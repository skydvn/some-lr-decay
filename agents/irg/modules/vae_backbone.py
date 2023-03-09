import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np
from utils.batchify import *
from agents.irg.modules.mlp_backbone import *
from agents.irg.modules.core import *
from agents.irg.modules.vaecore import VAEBaseLayer, _layer_init

# 4 players
class SimpleEncoder(VAEBaseLayer):
    def __init__(self, inchannel: int, outchannel: int) -> None:        
        super().__init__()
        self.network = nn.Sequential(
            # inchannel * 32 * 32
            _layer_init(nn.Conv2d(inchannel, 8, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 8 * 14 * 14
            _layer_init(nn.Conv2d(8, 16, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 16 * 5 * 5
            _layer_init(nn.Conv2d(16, 32, 4)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 32 * 1 * 1
        )
        self.outer = nn.Sequential(
            _layer_init(nn.Linear(32, outchannel)),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=(-2, -1))
        return self.outer(x)

class SimpleDecoder(VAEBaseLayer):
    def __init__(self, inchannel: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(inchannel, 32)),
            nn.Unflatten(1, (32, 1, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #32 * 2 * 2
            _layer_init(nn.ConvTranspose2d(32, 16, 4)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #16 * 10 * 10
            _layer_init(nn.ConvTranspose2d(16, 8, 5)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #8 * 28 * 28
            _layer_init(nn.ConvTranspose2d(8, 4, 5)),
            #4 * 32 * 32
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleEncoderSmall(VAEBaseLayer):
    def __init__(self, inchannel, outchannel: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # 4 * 32 * 64
            _layer_init(nn.Conv2d(inchannel, 16, 8, 4)),
            nn.ReLU(),
            # 16 * 7 * 7
            _layer_init(nn.Conv2d(16, 32, 5, 2)),
            nn.ReLU(),
            # 32 * 2 * 2
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, outchannel),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleDecoderSmall(VAEBaseLayer):
    def __init__(self, inchannel:int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inchannel, 32 * 2 * 2),
            nn.Unflatten(1, (32, 2, 2)),
            _layer_init(nn.ConvTranspose2d(32, 16, 5, 2)),
            nn.ReLU(),
            _layer_init(nn.ConvTranspose2d(16, 4, 8, 4)),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

# 2 players

class SimpleEncoder2Player(VAEBaseLayer):
    def __init__(self, inchannel: int, outchannel: int) -> None:        
        super().__init__()
        self.network = nn.Sequential(
            # inchannel * 32 * 64
            _layer_init(nn.Conv2d(inchannel, 8, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 8 * 14 * 30
            _layer_init(nn.Conv2d(8, 16, 5)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 16 * 5 * 13
            _layer_init(nn.Conv2d(16, 32, 4)),
            nn.MaxPool2d(2),
            nn.ReLU(),
            # 32 * 1 * 5
        )
        self.outer = nn.Sequential(
            _layer_init(nn.Linear(32, outchannel)),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=(-2, -1))
        return self.outer(x)

class SimpleDecoder2Player(VAEBaseLayer):
    def __init__(self, inchannel: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Linear(inchannel, 160)),
            nn.Unflatten(1, (32, 1, 5)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #32 * 2 * 2
            _layer_init(nn.ConvTranspose2d(32, 16, 4)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #16 * 10 * 10
            _layer_init(nn.ConvTranspose2d(16, 8, 5)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            #8 * 28 * 28
            _layer_init(nn.ConvTranspose2d(8, 4, 5)),
            #4 * 32 * 32
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleEncoder2PlayerSmall(VAEBaseLayer):
    def __init__(self, inchannel, outchannel: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            # 4 * 32 * 64
            _layer_init(nn.Conv2d(inchannel, 16, 8, 4)),
            nn.ReLU(),
            # 16 * 7 * 15
            _layer_init(nn.Conv2d(16, 32, 5, 2)),
            nn.ReLU(),
            # 32 * 2 * 6
            nn.Flatten(),
            nn.Linear(32 * 2 * 6, outchannel),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)

class SimpleDecoder2PlayerSmall(VAEBaseLayer):
    def __init__(self, inchannel:int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inchannel, 384),
            nn.Unflatten(1, (32, 2, 6)),
            _layer_init(nn.ConvTranspose2d(32, 16, 5, 2)),
            nn.ReLU(),
            _layer_init(nn.ConvTranspose2d(16, 4, 8, 4)),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.network(x)