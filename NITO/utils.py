import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import markers
from matplotlib.path import Path
from tqdm.autonotebook import trange

from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial import KDTree

class NITO_Dataset:
    def __init__(self, topologies, BCs, Cs, shapes, n_samples = 1024, noisy=False):
        
        self.topologies = topologies
        self.shapes = shapes
        self.n_samples = n_samples
        self.noisy = noisy
        
        self.BCs = BCs
        self.Cs = Cs
        
        self.n_BC = len(BCs)
        self.n_C = len(Cs)
        
        self.max_size = 0
        
        self.max_BC_size = np.zeros(len(self.BCs),dtype=int)
        
        for i in range(len(self.topologies)):
            top = self.topologies[i]
            
            if top.shape[0] > self.max_size:
                self.max_size = top.shape[0]
            
            for j in range(self.n_BC):
                if self.BCs[j][i].shape[0] > self.max_BC_size[j]:
                    self.max_BC_size[j] = self.BCs[j][i].shape[0]
        
    def __len__(self):
        return self.topologies.shape[0]
    
    def load(self, idx, mode='train'):
        topology = self.topologies[idx]
        BC = []
        C = []
        shape = self.shapes[idx]

        for i in range(self.n_BC):
            BC.append(self.BCs[i][idx])
        
        for i in range(self.n_C):
            C.append(self.Cs[i][idx])
    
        poses = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
        poses = np.concatenate([poses[1].reshape(-1,1),poses[0].reshape(-1,1)],axis=1)
        
        if mode == 'test':
            samples_idx = np.arange(poses.shape[0])
            
            pad_size = self.max_size - samples_idx.shape[0]
            additional_idx = np.random.choice(samples_idx,pad_size,replace=True)
            
            samples_idx = np.concatenate([samples_idx,additional_idx])
        else:
            if poses.shape[0] < self.n_samples:
                samples_idx = np.random.choice(poses.shape[0],self.n_samples,replace=True)
            else:
                samples_idx = np.random.choice(poses.shape[0],self.n_samples,replace=False)
        
        coords = poses[samples_idx]/shape.max()
        
        if self.noisy:
            coords += np.random.uniform(-1/shape.max()/2,1/shape.max()/2,coords.shape)
        
        labels = topology[samples_idx]
        
        return coords, labels, BC, C
    
    def batch_load(self, idxs, device=None, mode='train'):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        coords = []
        labels = []
        BCs = []
        BCs_mask = []
        Cs = []
        
        for i in range(self.n_BC):
            BCs.append([])
            BCs_mask.append([])
        
        for i in range(self.n_C):
            Cs.append([])
        
        max_bc_size = np.zeros(self.n_BC,dtype=int)
        
        for idx in idxs:
            coord,l,bc,c = self.load(idx, mode=mode)
            coords.append(coord)
            labels.append(l)
            mult = coord.shape[0]

            for i in range(self.n_BC):
                mask = np.zeros([self.max_BC_size[i],1],dtype=bool)
                mask[:bc[i].shape[0]] = 1
                bc_ = np.pad(bc[i],((0,self.max_BC_size[i]-bc[i].shape[0]),(0,0)))
                BCs[i].append(bc_)
                BCs_mask[i].append(mask)
                
                # if bc[i].shape[0] > max_bc_size[i]:
                #     max_bc_size[i] = bc[i].shape[0]
                
            
            for i in range(self.n_C):
                Cs[i].append(c[i])
        
        # for i in range(self.n_BC):
        #     BCs[i] = np.array(BCs[i])
        #     BCs_mask[i] = np.array(BCs_mask[i])
        #     BCs[i] = BCs[i][:,0:max_bc_size[i],:]
        #     BCs_mask[i] = BCs_mask[i][:,0:max_bc_size[i],:]
        
        coords = np.concatenate(coords,0)
        labels = np.concatenate(labels,0)

        coords = torch.tensor(coords).float().to(device)
        labels = torch.tensor(labels).float().to(device)

        # B_BC = []
        # for i in range(self.n_BC):
        #     B_BC.append([])

        # for i in range(len(idxs)):
        #     for j in range(self.n_BC):
        #         B_BC[j].append(BCs[j][i].shape[0])
        
        # for i in range(self.n_BC):
        #     B_BC[i] = np.repeat(np.arange(len(idxs)),B_BC[i])
        #     B_BC[i] = torch.tensor(B_BC[i]).long().to(device)
        
        for i in range(self.n_BC):
            BCs[i] = np.concatenate(BCs[i],0)
            BCs[i] = torch.tensor(BCs[i]).float().to(device)
            
            BCs_mask[i] = np.concatenate(BCs_mask[i],0)
            BCs_mask[i] = torch.tensor(BCs_mask[i]).bool().to(device)

        for i in range(self.n_C):
            Cs[i] = np.array(Cs[i])
            if len(Cs[i].shape) == 1:
                Cs[i] = Cs[i].reshape(-1,1)
            Cs[i] = torch.tensor(Cs[i]).float().to(device)
        
        inputs = [coords, mult, BCs, BCs_mask, Cs, idxs.shape[0]]
        
        return inputs, labels