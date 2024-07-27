import argparse
import os
import torch
import numpy as np
import pickle

from NITO.utils import NITO_Dataset
from NITO.model import NITO
from NITO.trainer import Trainer

# setup arguments
parser = argparse.ArgumentParser(description='NITO Training Arguments')
parser.add_argument('--data', type=str, default='./Data', help='path to data directory. Default: ./Data')
parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint file to load. Default: None')
parser.add_argument('--checkpoint_dir', type=str, default='./Checkpoints', help='path to checkpoint directory. Default: ./Checkpoints')
parser.add_argument('--checkpoint_freq', type=int, default=5, help='checkpoint frequency (epochs). Default: 5')
parser.add_argument('--name', type=str, default='NITO', help='name of the model. Default: NITO')
parser.add_argument('--batch_size', type=int, default=64, help='batch size. Default: 64')
parser.add_argument('--samples', type=int, default=1024, help='number of points sampled per topology. Default: 1024')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs. Default: 100')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. Default: 1e-4')

# model arguments
parser.add_argument('--BC_n_layers', type=int, default=4, help='number of layers in BC encoder. Default: 4')
parser.add_argument('--BC_hidden_size', type=int, default=256, help='hidden size of BC encoder. Default: 256')
parser.add_argument('--BC_emb_size', type=int, default=64, help='embedding size of BC encoder (3x). Default: 64')
parser.add_argument('--C_n_layers', type=int, default=4, help='number of layers in C encoder. Default: 4')
parser.add_argument('--C_hidden_size', type=int, default=256, help='hidden size of C encoder. Default: 256')
parser.add_argument('--C_mapping_size', type=int, default=256, help='mapping size of C encoder. Default: 256')
parser.add_argument('--Field_n_layers', type=int, default=12, help='number of layers in field network. Default: 12')
parser.add_argument('--Field_hidden_size', type=int, default=6144, help='hidden size of field network. Default: 6144')
parser.add_argument('--Fourier_size', type=int, default=512, help='size of Fourier features. Default: 512')
parser.add_argument('--omega', type=float, default=1.0, help='omega value. Default: 1.0')
parser.add_argument('--freq_scale', type=float, default=10.0, help='frequency scale. Default: 10.0')

args = parser.parse_args()

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

# create dataset
dataset = NITO_Dataset(topologies, [BCs, loads], [vfs, shapes], shapes, n_samples=args.samples)


if args.checkpoint is not None:
    with open(args.checkpoint, 'rb') as f:
        trainer = pickle.load(f)
    # parameter count
    print('Model Parameters:', sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))

else:
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

    # parameter count
    print('Model Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # create trainer
    trainer = Trainer(model,
                    lr=args.lr,
                    schedule_max_steps=args.epochs)

    # save initial model
    with open(os.path.join(args.checkpoint_dir, f'{args.name}_0.NITO'), 'wb') as f:
        pickle.dump(trainer, f)

for i in range(args.epochs):
    trainer.train(dataset.batch_load, np.arange(len(dataset))[0:-5000], args.batch_size, epochs=1)
    if (i+1) % args.checkpoint_freq == 0:
        with open(os.path.join(args.checkpoint_dir, f'{args.name}_{i+1}.NITO'), 'wb') as f:
            pickle.dump(trainer, f)