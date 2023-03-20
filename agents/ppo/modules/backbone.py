import os, sys
sys.path.append(os.getcwd())
from agents.ppo.modules.core import *

import numpy as np
import torch.nn as nn

class ActorCriticSiamese(ACSiamese):
    def __init__(self, num_actions: int, stack_size: int) -> nn.Module:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Conv2d(stack_size, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions)),
        )
        self.critic = nn.Sequential(
            nn.Conv2d(stack_size, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1))
        )

class ActorCriticSiameseSmall(ACSiamese):
    def __init__(self, num_actions: int, stack_size: int) -> nn.Module:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Conv2d(stack_size, 32, 8, stride=4),
            # 1 * 32 * 15 * 15
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            # 1 * 64 * 6 * 6
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            # 1 * 64 * 4 * 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, num_actions)),
        )
        self.critic = nn.Sequential(
            nn.Conv2d(stack_size, 32, 8, stride=4),
            # 1 * 32 * 15 * 15
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            # 1 * 64 * 6 * 6
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            # 1 * 64 * 4 * 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1))
        )

class ActorCriticSiameseNano(ACSiamese):
    def __init__(self, num_actions: int, stack_size: int) -> nn.Module:
        super().__init__()
        self.actor = nn.Sequential(
            # 4 * 32 * 64
            nn.Conv2d(stack_size, 16, 8, 4),
            nn.ReLU(),
            # 16 * 7 * 15
            nn.Conv2d(16, 32, 5, 2),
            nn.ReLU(),
            # 32 * 2 * 6
            nn.Flatten(),
            nn.Linear(32 * 2 * 6, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, num_actions))
        )

        self.critic = nn.Sequential(
            # 4 * 32 * 64
            nn.Conv2d(stack_size, 16, 8, 4),
            nn.ReLU(),
            # 16 * 7 * 15
            nn.Conv2d(16, 32, 5, 2),
            nn.ReLU(),
            # 32 * 2 * 6
            nn.Flatten(),
            nn.Linear(32 * 2 * 6, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1))
        )

class ActorCriticMultiHead(ACMultiHead):
    def __init__(self, num_actions: int, stack_size: int) -> nn.Module:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(stack_size, 32, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, num_actions))
        )
        self.critic = layer_init(nn.Linear(512, 1))
            

class ActorCriticMultiHeadSmall(ACMultiHead):
    def __init__(self, num_actions: int, stack_size: int) -> nn.Module:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(stack_size, 32, 8, stride=4),
            # 1 * 32 * 15 * 15
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            # 1 * 64 * 6 * 6
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            # 1 * 64 * 4 * 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(512, num_actions))
        )
        self.critic = layer_init(nn.Linear(512, 1))