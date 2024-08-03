import argparse
import os
import torch
import numpy as np
import pickle

from NITO.utils import NITO_Dataset
from NITO.model import NITO
from NITO.trainer import Trainer

from torch.profiler import profile, record_function, ProfilerActivity
import uuid
# setup arguments
parser = argparse.ArgumentParser(description='NITO Training Arguments')
parser.add_argument('--data', type=str, default='./Data', help='path to data directory. Default: ./Data')
parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint file to load. Default: None')
parser.add_argument('--checkpoint_dir', type=str, default='./Checkpoints', help='path to checkpoint directory. Default: ./Checkpoints')
parser.add_argument('--checkpoint_freq', type=int, default=5, help='checkpoint frequency (epochs). NOTE: every epoch will be checkpointed but they will be replaced evey epoch and only the checkpoints of this frequency will be kept. Default: 5')
parser.add_argument('--name', type=str, default='NITO', help='name of the model. Default: NITO')
parser.add_argument('--batch_size', type=int, default=32, help='batch size. Default: 32')
parser.add_argument('--samples', type=int, default=1024, help='number of points sampled per topology. Default: 1024')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs. Default: 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. Default: 1e-4')
parser.add_argument('--multi_gpu', action='store_true', help='Data parallel training. Default: False')
parser.add_argument('--mixed_precision', action='store_true', help='Mixed precision training. Default: False')
parser.add_argument('--DDP', action='store_true', help='Distributed data parallel training. Default: False')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model before training. Recommended to set to true. Default: False')
parser.add_argument('--supress_warnings', action='store_true', help='Supress warnings. Default: False')
parser.add_argument('--profile', action='store_true', help='Profile the model. NOTE: This will do only 5 steps and no checkpointing. Purely for memory profiling. Default: False')

# model arguments
parser.add_argument('--BC_n_layers', type=int, default=4, help='number of layers in BC encoder. Default: 4')
parser.add_argument('--BC_hidden_size', type=int, default=256, help='hidden size of BC encoder. Default: 256')
parser.add_argument('--BC_emb_size', type=int, default=64, help='embedding size of BC encoder (3x). Default: 64')
parser.add_argument('--C_n_layers', type=int, default=4, help='number of layers in C encoder. Default: 4')
parser.add_argument('--C_hidden_size', type=int, default=256, help='hidden size of C encoder. Default: 256')
parser.add_argument('--C_mapping_size', type=int, default=256, help='mapping size of C encoder. Default: 256')
parser.add_argument('--Field_n_layers', type=int, default=8, help='number of layers in field network. Default: 8')
parser.add_argument('--Field_hidden_size', type=int, default=2048, help='hidden size of field network. Default: 2048')
parser.add_argument('--Fourier_size', type=int, default=512, help='size of Fourier features. Default: 512')
parser.add_argument('--omega', type=float, default=1.0, help='omega value. Default: 1.0')
parser.add_argument('--freq_scale', type=float, default=10.0, help='frequency scale. Default: 10.0')

args = parser.parse_args()

if args.supress_warnings:
    import warnings
    warnings.filterwarnings("ignore")

if args.DDP and "WORLD_SIZE" not in os.environ:
    raise ValueError("Error: DDP flag is set, but script is not launched with torchrun. Please use: torchrun --nproc_per_node=NUM_GPUS train.py --DDP [other args]")

# make checkpoint directory if it does not exist
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

# load data
if not os.path.exists(args.data):
    raise ValueError('Data directory does not exist')

topologies = np.load(os.path.join(args.data, 'topologies.npy'), allow_pickle=True)
shapes = np.load(os.path.join(args.data, 'shapes.npy'), allow_pickle=True)
loads = np.load(os.path.join(args.data, 'loads.npy'), allow_pickle=True)
vfs = np.load(os.path.join(args.data, 'vfs.npy'), allow_pickle=True)
BCs = np.load(os.path.join(args.data, 'boundary_conditions.npy'), allow_pickle=True)

if (args.DDP and args.multi_gpu) or not args.multi_gpu:
    consistent_batch = False
else:
    consistent_batch = True

# create dataset
dataset = NITO_Dataset(topologies, [BCs, loads], [vfs, shapes], shapes, n_samples=args.samples, consistent_batch=consistent_batch)

# create model
model = NITO(BCs = [4,4],
            BC_n_layers = [args.BC_n_layers,args.BC_n_layers],
            BC_hidden_size = [args.BC_hidden_size,args.BC_hidden_size], 
            BC_emb_size=[args.BC_emb_size,args.BC_emb_size], 
            Cs = [1,2],
            C_n_layers = [args.C_n_layers,args.C_n_layers],
            C_hidden_size = [args.C_hidden_size,args.C_hidden_size],
            C_mapping_size = [args.C_mapping_size,args.C_mapping_size],
            Field_n_layers=args.Field_n_layers, 
            Field_hidden_size=args.Field_hidden_size, 
            Fourier_size=args.Fourier_size, 
            omega = args.omega,
            freq_scale= args.freq_scale)

# create trainer
trainer = Trainer(model,
                lr=args.lr,
                schedule_max_steps=args.epochs,
                multi_gpu=args.multi_gpu,
                mixed_precision=args.mixed_precision,
                DDP_train = args.DDP,
                checkpoint_path=args.checkpoint,
                Compile=args.compile,
                enable_profiling=args.profile)

# parameter count
if trainer.is_main_process():
    print('Model Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

if not args.profile:
    trainer.save_checkpoint(os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{trainer.current_epoch}.pth'))

    trainer.train(dataset.batch_load, np.arange(len(dataset))[0:-5000], args.batch_size, epochs=args.epochs, checkpoint_dir=args.checkpoint_dir, checkpoint_interval=args.checkpoint_freq)             
else:
    if trainer.DDP:
        rank = trainer.rank
        context = f'rank_{rank}'
    else:
        context = 'single_gpu'
        
    prof = torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/Training_Profile_{uuid.uuid4()}_{context}'),
        record_shapes=True,
        with_stack=True,
        profile_memory=True)
    
    prof.start()
    trainer.profile(dataset.batch_load, np.arange(len(dataset))[0:-5000], args.batch_size)
    prof.stop()

    prof.export_memory_timeline(f"mem_{context}.html")