import argparse
import os
import torch
import numpy as np
import pickle
import warnings

from NITO.utils import NITO_Dataset
from NITO.model import NITO

from ATOMS.geometry import generate_structured_mesh
from ATOMS.utils import filter_2D_structured, filter_3D_structured
from ATOMS.MaterialModels import SingleMaterial, PenalizedMultiMaterial
from ATOMS.solver import Solver

from joblib import Parallel, delayed, cpu_count

from tqdm.autonotebook import trange

import tabulate
import time

# setup arguments
parser = argparse.ArgumentParser(description='NITO Evaluation Arguments')
parser.add_argument('--data', type=str, default='./Data', help='path to data directory. Default: ./Data')
parser.add_argument('--start_index', type=int, default=0, help='start index of data. Default: 0')
parser.add_argument('--end_index', type=int, default=None, help='end index of data. Default: None')
parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint file to load. Default: None')
parser.add_argument('--results_dir', type=str, default='./Results', help='path to save results in. Default: ./Results')
parser.add_argument('--mixed_precision', action='store_true', help='Mixed precision training. Default: False')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model before training. Recommended to set to true. Default: False')
parser.add_argument('--supress_warnings', action='store_true', help='Supress warnings. Default: False')
parser.add_argument('--shape_normalize', action='store_true', help='Normalize shapes. Default: False')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use. Default: 0')
parser.add_argument('--batch_size', type=int, default=1, help='batch size. Default: 1')
parser.add_argument('--mixed_resolution', action='store_true', help='Mixed resolution evaluation with batch larger than 1. Default: False')
parser.add_argument('--multi_processing', action='store_true', help='Use multi processing for direct optimization evaluation. Default: False')
parser.add_argument('--n_jobs', type=int, default=cpu_count(), help='Number of jobs for multi processing. Default: cpu_count()')
parser.add_argument('--time_inference', action='store_true', help='Time inference. Default: False')
parser.add_argument('--n_time_experiments', type=int, default=100, help='Number of experiments for timing inference. Default: 100')
parser.add_argument('--ignore_outliers', action='store_true', help='Ignore outliers. Default: False')
parser.add_argument('--outlier_CE_threshold', type=float, default=10, help='Outlier threshold for CE. Default: 10, ie 1000%')

# load pre-computed
parser.add_argument('--load_raw_from', type=str, default=None, help='Load raw model outputs from this numpy file. Default: None')

# what to save
parser.add_argument('--save_raw', action='store_true', help='Save raw model outputs. Default: False')
parser.add_argument('--save_intermediate', action='store_true', help='Save intermediate predcictions with fewer direct optimization. Default: False')
parser.add_argument('--save_optimized', action='store_true', help='Save optimized predictions. Default: False')
parser.add_argument('--save_compliances', action='store_true', help='Save compliances values. Default: False')
parser.add_argument('--save_CE', action='store_true', help='Save compliance error values. Default: False')
parser.add_argument('--save_VFE', action='store_true', help='Save volume fraction error values. Default: False')

# direct optimizer setup
parser.add_argument('--precompute_kernel', action='store_true', help='Precompute kernel. Default: False')
parser.add_argument('--do_penalty', type=float, default=3, help='Penalty value. Default: 3.0')
parser.add_argument('--do_move', type=float, default=0.2, help='Move value for SIMP. Default: 0.2')
parser.add_argument('--do_steps', type=int, default=10, help='Number of steps for direct optimization. Default: 10')
parser.add_argument('--do_intermidiate_steps', type=int, default=5, help='Number of steps for direct optimization checkpoint. Default: 5')

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

# disable gradient computation
torch.set_grad_enabled(False)

if args.supress_warnings:
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")

# make results directory if it does not exist
if not os.path.exists(args.results_dir):
    os.makedirs(args.results_dir)

# if no checkpoint is provided, raise error
if args.checkpoint is None:
    raise ValueError("Error: No checkpoint provided. Please provide a checkpoint to evaluate the model.")

# load data
if not os.path.exists(args.data):
    raise ValueError('Data directory does not exist')

topologies = np.load(os.path.join(args.data, 'topologies.npy'), allow_pickle=True)
shapes = np.load(os.path.join(args.data, 'shapes.npy'), allow_pickle=True)
loads = np.load(os.path.join(args.data, 'loads.npy'), allow_pickle=True)
vfs = np.load(os.path.join(args.data, 'vfs.npy'), allow_pickle=True)
BCs = np.load(os.path.join(args.data, 'boundary_conditions.npy'), allow_pickle=True)

# create dataset
if args.shape_normalize:
    dataset = NITO_Dataset(topologies, [BCs, loads], [vfs, shapes/shapes.max(1,keepdims=True)], shapes, n_samples=10, consistent_batch=False)
else:
    dataset = NITO_Dataset(topologies, [BCs, loads], [vfs, shapes], shapes, n_samples=10, consistent_batch=False)

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

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
            freq_scale= args.freq_scale).to(device)

# load checkpoint
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

if args.compile:
    model.compile()

# compute predictions
start_index = args.start_index
end_index = args.end_index if args.end_index is not None else len(topologies)
indecies = np.arange(start_index,end_index)

if args.load_raw_from is not None:
    preds = np.load(args.load_raw_from, allow_pickle=True)
    
else:
    pred_tops = []

    if args.batch_size > 1 and args.mixed_resolution:
        mode = 'test'
    else:
        mode = 'test_no_pad'
    
    n_steps = int(np.ceil(len(indecies)/args.batch_size))
    progress = trange(n_steps)

    if args.mixed_precision:
        with torch.cuda.amp.autocast():
            for i in progress:
                rindx = indecies[i*args.batch_size:(i+1)*args.batch_size]
                inps, _ = dataset.batch_load(rindx,device,mode=mode)
                labels = torch.sigmoid(model(inps)).reshape(-1)
                pred_tops.append(labels.detach().cpu().numpy())
    else:
        for i in progress:
            rindx = indecies[i*args.batch_size:(i+1)*args.batch_size]
            inps, _ = dataset.batch_load(rindx,device,mode=mode)
            labels = torch.sigmoid(model(inps)).reshape(-1)
            pred_tops.append(labels.detach().cpu().numpy())

    pred_tops = np.concatenate(pred_tops)

    preds = []

    if mode == 'test_no_pad':
        curent_pointer = 0
        for i in indecies:
            pred = pred_tops[curent_pointer:curent_pointer+shapes[i].prod()]
            preds.append(pred)
            curent_pointer += shapes[i].prod()
    else:
        for i, id in enumerate(indecies):
            pred = pred_tops[i*dataset.max_size:i*dataset.max_size+shapes[id].prod()]
            preds.append(pred)

    preds = np.array(preds,dtype=object)

    if args.save_raw:
        np.save(os.path.join(args.results_dir,'raw.npy'),preds, allow_pickle=True)


# direct optimization
if args.precompute_kernel:
    solvers = {}

    unique_shapes = np.unique(shapes[indecies],axis=0)

    for i,shape in enumerate(unique_shapes):
        elements, nodes = generate_structured_mesh(dim=shape/shape.max(), nel=shape)
        filter_kernel = filter_2D_structured(elements=elements, nodes=nodes, nelx = shape[0], nely = shape[1], r_min = 1.5 * 1/shape.max())
        material = SingleMaterial(E=1, nu=0.33, penalty=3, volume_fraction= 0.5, void=1e-9)
        solver = Solver(mesh=(nodes,elements),filter_kernel=filter_kernel,material_model=material,structured=True)
        solvers[str(shape)] = solver
else:
    solvers = None

def evaluate_with_directopt(top, BC, pred_top, shape, load, vf, move=0.2, pen=3, n_1=5, n_2=10, solver=None):
    if solver is None:
        elements, nodes = generate_structured_mesh(dim=shape/shape.max(), nel=shape)
        filter_kernel = filter_2D_structured(elements=elements, nodes=nodes, nelx = shape[0], nely = shape[1], r_min = 1.5 * 1/shape.max())
        solver = Solver(mesh=(nodes,elements),filter_kernel=filter_kernel,structured=True)
    
    mat_model = SingleMaterial(E=1, nu=0.33, penalty=pen, volume_fraction= vf, void=1e-9)
    solver.material_model = mat_model
    solver.move = move
    
    solver.reset_BC()
    solver.reset_F()
    solver.add_BCs(BC[:,0:2], BC[:,2:])
    solver.add_Forces(load[:,0:2], load[:,2:])

    pred_top = np.array(pred_top.reshape(-1,1),dtype=float)

    pred_2,pred_1,_ = solver.fs_optimize(rho=pred_top, n_step = n_2, chk_steps = n_1)

    comp_2 = solver.FEA(rho=pred_2>0.5)[1]
    comp_1 = solver.FEA(rho=pred_1>0.5)[1]
    comp_0 = solver.FEA(rho=pred_top>0.5)[1]
    comp_gt = solver.FEA(rho=top.reshape(-1,1))[1]

    vf_2 = np.sum(pred_2>0.5)/shape.prod()
    vf_1 = np.sum(pred_1>0.5)/shape.prod()
    vf_0 = np.sum(pred_top>0.5)/shape.prod()

    CE_2 = (comp_2 - comp_gt)/comp_gt
    CE_1 = (comp_1 - comp_gt)/comp_gt
    CE_0 = (comp_0 - comp_gt)/comp_gt

    VFE_2 = abs(vf_2 - vf)/vf
    VFE_1 = abs(vf_1 - vf)/vf
    VFE_0 = abs(vf_0 - vf)/vf

    return [CE_2,CE_1,CE_0], [VFE_2,VFE_1,VFE_0], [comp_2,comp_1,comp_0,comp_gt], [pred_2,pred_1]

def inference_with_do(inputs, model, BC, shape, load, vf, move=0.2, pen=3, n_1=5, n_2=10, solver=None):
    if solver is None:
        elements, nodes = generate_structured_mesh(dim=shape/shape.max(), nel=shape)
        filter_kernel = filter_2D_structured(elements=elements, nodes=nodes, nelx = shape[0], nely = shape[1], r_min = 1.5 * 1/shape.max())
        solver = Solver(mesh=(nodes,elements),filter_kernel=filter_kernel,structured=True)
    
    mat_model = SingleMaterial(E=1, nu=0.33, penalty=pen, volume_fraction= vf, void=1e-9)
    solver.material_model = mat_model
    solver.move = move
    
    solver.reset_BC()
    solver.reset_F()
    solver.add_BCs(BC[:,0:2], BC[:,2:])
    solver.add_Forces(load[:,0:2], load[:,2:])
    
    labels = torch.sigmoid(model(inputs)).reshape(-1)
    pred_top = labels.detach().cpu().numpy()

    pred_top = np.array(pred_top.reshape(-1,1),dtype=float)

    pred_2,pred_1,_ = solver.fs_optimize(rho=pred_top, n_step = n_2, chk_steps = n_1)

def run_inference(i,  move=0.2, pen=3, n_1=5, n_2=10, solvers=None):
    idx = indecies[i]
    top = topologies[idx]
    shape = shapes[idx]
    load = loads[idx]
    vf = vfs[idx]
    BC = BCs[idx]
    inputs, _ = dataset.batch_load([idx],device,mode='test_no_pad')
    if solvers is not None:
        solver = solvers[str(shape)]
    else:
        solver = None
    return inference_with_do(inputs, model, BC, shape, load, vf, move=move, pen=pen, n_1=n_1, n_2=n_2, solver=solver)

def run_do_eval(i, move=0.2, pen=3, n_1=5, n_2=10, solvers=None):
    idx = indecies[i]
    top = topologies[idx]
    shape = shapes[idx]
    load = loads[idx]
    vf = vfs[idx]
    BC = BCs[idx]
    pred_top = preds[i]
    if solvers is not None:
        solver = solvers[str(shape)]
    else:
        solver = None
    return evaluate_with_directopt(top, BC, pred_top, shape, load, vf, solver=solver, move=move, pen=pen, n_1=n_1, n_2=n_2)

if args.multi_processing:
    results = Parallel(n_jobs=args.n_jobs)(delayed(run_do_eval)(idx, move=args.do_move, pen=args.do_penalty, n_1=args.do_intermidiate_steps, n_2=args.do_steps, solvers=solvers) for idx in trange(len(indecies)))
else:
    results = []
    for idx in trange(len(indecies)):
        results.append(run_do_eval(idx, move=args.do_move, pen=args.do_penalty, n_1=args.do_intermidiate_steps, n_2=args.do_steps, solvers=solvers))

CE = np.zeros((len(indecies),3))
VFE = np.zeros((len(indecies),3))
compliances = np.zeros((len(indecies),4))

pred_1 = []
pred_2 = []

for i in range(len(indecies)):
    CE[i] = np.array(results[i][0])
    VFE[i] = np.array(results[i][1])
    compliances[i] = np.array(results[i][2])
    pred_2.append(results[i][3][0])
    pred_1.append(results[i][3][1])

if args.save_optimized:
    np.save(os.path.join(args.results_dir,'optimized.npy'),np.array(pred_2,dtype=object), allow_pickle=True)

if args.save_intermediate:
    np.save(os.path.join(args.results_dir,'intermediate.npy'),np.array(pred_1,dtype=object), allow_pickle=True)

if args.save_compliances:
    np.save(os.path.join(args.results_dir,'compliances.npy'),compliances, allow_pickle=True)

if args.save_CE:
    np.save(os.path.join(args.results_dir,'CE.npy'),CE, allow_pickle=True)

if args.save_VFE:
    np.save(os.path.join(args.results_dir,'VFE.npy'),VFE, allow_pickle=True)

if args.ignore_outliers:
    usables = np.all(np.abs(CE) < args.outlier_CE_threshold, axis=1)
    print('Outlier Count:',len(indecies) - np.sum(usables))

    CE = CE[usables]
    VFE = VFE[usables]
    compliances = compliances[usables]

if args.time_inference:
    print('Timing Inference...')
    inference_times = np.zeros(args.n_time_experiments)
    if args.mixed_precision:
        with torch.cuda.amp.autocast():
            for i in trange(args.n_time_experiments+5):
                start = time.time()
                idx = np.random.randint(len(indecies))
                run_inference(idx, move = args.do_move, pen=args.do_penalty, n_1=args.do_intermidiate_steps, n_2=args.do_steps, solvers=solvers)
                if i>=5:
                    inference_times[i-5] = time.time() - start
    else:
        for i in range(args.n_time_experiments + 5):
            start = time.time()
            idx = np.random.randint(len(indecies))
            run_inference(idx, move = args.do_move, pen=args.do_penalty, n_1=args.do_intermidiate_steps, n_2=args.do_steps, solvers=solvers)
            if i>=5:
                inference_times[i-5] = time.time() - start
    inference_times = np.array(inference_times)

    print('Inference Time:',np.mean(inference_times), '±', np.std(inference_times))

# pretty print a table summary
headers = ['Model', 'CE (%)', 'CE Median (%)', 'VFE (%)', 'VFE Median (%)']

CE_median = np.median(CE,axis=0) * 100
CE_mean = np.mean(CE,axis=0) * 100
CE_std = np.std(CE,axis=0) * 100
VFE_median = np.median(VFE,axis=0) * 100
VFE_mean = np.mean(VFE,axis=0) * 100
VFE_std = np.std(VFE,axis=0) * 100

data = [
    ['Neural Field', f'{CE_mean[2]:.3f} ± {CE_std[2]:.3f}', f'{CE_median[2]:.3f}', f'{VFE_mean[2]:.3f} ± {VFE_std[2]:.3f}', f'{VFE_median[2]:.3f}'],
    [f'NITO ({args.do_intermidiate_steps})', f'{CE_mean[1]:.3f} ± {CE_std[1]:.3f}', f'{CE_median[1]:.3f}', f'{VFE_mean[1]:.3f} ± {VFE_std[1]:.3f}', f'{VFE_median[1]:.3f}'],
    [f'NITO ({args.do_steps})', f'{CE_mean[0]:.3f} ± {CE_std[0]:.3f}', f'{CE_median[0]:.3f}', f'{VFE_mean[0]:.3f} ± {VFE_std[0]:.3f}', f'{VFE_median[0]:.3f}']
]

print(f'Results Summary For {len(indecies)} Topologies :')
print(tabulate.tabulate(data, headers=headers, tablefmt='fancy_grid'))