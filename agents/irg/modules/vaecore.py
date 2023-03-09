import torch
import numpy as np
from torch import nn

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.kaiming_uniform_(layer.weight) #he normal
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class VAEBaseLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        raise NotImplementedError
    
    def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.kaiming_uniform_(layer.weight) #he normal
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer