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