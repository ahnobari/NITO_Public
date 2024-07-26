import torch

from torch import nn
from torch.nn import functional as F

class BC_Encoder(nn.Module):
    def __init__(self, mlp_layers):
        super(BC_Encoder, self).__init__()
        self.mlp = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            # Add Layer Normalization except for the last layer
            if i < len(mlp_layers) - 2:
                self.layer_norms.append(nn.LayerNorm(mlp_layers[i+1]))

    def forward(self, positions, batch_index):
        # Apply MLP with Layer Normalization and ReLU to positions
        x = positions
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:  # Apply normalization and ReLU except for the last layer
                x = self.layer_norms[i](x)
                x = F.relu(x)

        # Enhanced pooling - mean, max, and min pooling
        batch_size = int(batch_index.max().item() + 1)
        pooled = []
        for index in range(batch_size):
            mask = (batch_index == index).squeeze()
            batch_x = x[mask]

            mean_pool = batch_x.mean(dim=0).squeeze()
            max_pool = batch_x.max(dim=0)[0].squeeze()
            min_pool = batch_x.min(dim=0)[0].squeeze()

            # Concatenate pooled features
            pooled_features = torch.cat((mean_pool, max_pool, min_pool), dim=0)
            pooled.append(pooled_features)

        # Stack pooled outputs for each set in the batch
        output = torch.stack(pooled)

        return output
    
class C_Encoder(nn.Module):
    def __init__(self, mlp_layers):
        super(C_Encoder, self).__init__()
        self.mlp = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(len(mlp_layers) - 1):
            self.mlp.append(nn.Linear(mlp_layers[i], mlp_layers[i+1]))
            # Add Layer Normalization except for the last layer
            if i < len(mlp_layers) - 2:
                self.layer_norms.append(nn.LayerNorm(mlp_layers[i+1]))

    def forward(self, inputs):
        # Apply MLP with Layer Normalization and ReLU to inputs
        x = inputs
        for i, layer in enumerate(self.mlp):
            x = layer(x)
            if i < len(self.mlp) - 1:
                x = self.layer_norms[i](x)
                x = F.relu(x)
        return x
    
class ConditionalLayerNorm1D(nn.Module):
    def __init__(self, in_dim, condition_dim):
        super().__init__()
        self.in_dim = in_dim
        
        self.c_mean_shift = nn.Linear(condition_dim,in_dim)
        self.c_var_mult = nn.Linear(condition_dim,in_dim)
    
    def forward(self, inputs, conditions):

        weights = self.c_var_mult(conditions)
        biases = self.c_mean_shift(conditions)

        return torch.nn.functional.layer_norm(inputs,[self.in_dim]) * weights + biases
    
class FourierConditionalNeuralField(nn.Module):
    def __init__(self, input_channels, num_layers, num_channels, condition_dim, output_channels, mapping_size, omega = 1.0, freq_scale = 10.0):
        super(FourierConditionalNeuralField, self).__init__()
        
        self.B = torch.nn.Parameter(torch.randn(size=(1,mapping_size,input_channels)) * freq_scale, requires_grad=True)

        # Define network layers
        self.layers = nn.ModuleList([nn.Linear(mapping_size * 2, num_channels)])  # First layer takes mapped coords
        for _ in range(1, num_layers - 1):
            self.layers.append(ConditionalLayerNorm1D(num_channels, condition_dim))
            self.layers.append(nn.Linear(num_channels, num_channels))
        
        self.layers.append(ConditionalLayerNorm1D(num_channels, condition_dim))

        # Output layer
        self.output_layer = nn.Linear(num_channels, output_channels)
        self.output_activation = nn.Sigmoid()

        self.omega = omega

    def fourier_feature_mapping(self, x, B):
        x_proj = torch.bmm((2.0 * torch.pi * x).unsqueeze(1),B.transpose(1,2)).squeeze()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x, conditions):

        B  = self.B.tile([x.shape[0],1,1])
        
        # Apply Fourier feature mapping
        x = self.fourier_feature_mapping(x,B)

        # Process through network layers
        for layer in self.layers:
            if isinstance(layer, ConditionalLayerNorm1D):
                x = layer(x, conditions)
                x = torch.sin(x * self.omega)
            else:
                x = layer(x)
        return self.output_layer(x)