import torch
from torch import nn
import numpy as np
from typing import Dict, List, Tuple, Type, Union, Sequence, Callable, Optional, Any
from agents.irg.modules.core import BaseLayer, _layer_init

class MLP(BaseLayer):
    def __init__(self, channels: List[int], 
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
        last_layer_activation: bool = False,
        device = "cuda") -> None:
        super().__init__()
        self.last_layer_activation = last_layer_activation
        self.activation_fn = activation_layer
        self.channels = channels
        self.device = device
        self.layers = []
        for i in range(len(self.channels)-1):
            self.layers.append(_layer_init(nn.Linear(self.channels[i], self.channels[i+1])).to(self.device))
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.channels) or self.last_layer_activation:
                x = self.activation_fn(x)
        return x