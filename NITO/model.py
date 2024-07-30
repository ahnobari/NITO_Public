from .nn import BC_Encoder, C_Encoder, FourierConditionalNeuralField

import torch
from torch import nn
from torch.nn import functional as F


class NITO(nn.Module):
    def __init__(self, output_channels = 1,
                       input_channels = 2,
                       BCs = [4,4],
                       BC_n_layers = [4,4],
                       BC_hidden_size = [256,256], 
                       BC_emb_size=[64,64], 
                       Cs = [1,2],
                       C_n_layers = [1,1],
                       C_hidden_size = [256,256],
                       C_mapping_size = [256,256],
                       Field_n_layers=8, 
                       Field_hidden_size=1024, 
                       Fourier_size=256, 
                       omega = 1.0,
                       freq_scale= 10.0):
        super().__init__()
        
        # check if BC_n_layers and BC_hidden_size are lists of same length
        assert len(BC_n_layers) == len(BC_hidden_size)
        assert len(BC_n_layers) == len(BC_emb_size)
        assert len(BC_n_layers) == len(BCs)

        # check if C_mapping_size is a list of same length as Cs
        assert len(C_n_layers) == len(Cs)
        assert len(C_n_layers) == len(C_mapping_size)
        assert len(C_n_layers) == len(C_hidden_size)
        
        
        self.BC_Networks = nn.ModuleList()
        for i in range(len(BC_n_layers)):
            self.BC_Networks.append(BC_Encoder([BCs[i]] + [BC_hidden_size[i]]* BC_n_layers[i] + [BC_emb_size[i]]))
        
        self.C_Networks = nn.ModuleList()
        for i in range(len(C_mapping_size)):
            self.C_Networks.append(C_Encoder([Cs[i]] + [C_hidden_size[i]]* C_n_layers[i] + [C_mapping_size[i]]))

        self.gen = FourierConditionalNeuralField(
                                                 input_channels,
                                                 Field_n_layers,
                                                 Field_hidden_size,
                                                 sum(BC_emb_size) * 3 + sum(C_mapping_size),
                                                 output_channels,
                                                 Fourier_size,
                                                 omega,
                                                 freq_scale)


    def forward(self, inputs):
        coords, mult, BCs, BC_mask, Cs, bc_size = inputs
        
        BC_emb = []
        for i in range(len(BCs)):
            BC_emb.append(self.BC_Networks[i](BCs[i],BC_mask[i],bc_size[i]))
        
        BC_emb = torch.cat(BC_emb,-1)
        
        C_emb = []
        for i in range(len(Cs)):
            C_emb.append(self.C_Networks[i](Cs[i]))
        
        C_emb = torch.cat(C_emb,-1)
        
        conditions = torch.cat([BC_emb,C_emb],-1).unsqueeze(1).tile([1,mult,1])
        conditions = conditions.view(-1,conditions.shape[-1])

        x = coords
        
        out = self.gen(x,conditions)
        
        return out