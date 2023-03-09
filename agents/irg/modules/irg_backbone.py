import os, sys
sys.path.append(os.getcwd())
import torch
from torch import nn
import numpy as np
from utils.batchify import *
from agents.irg.modules.mlp_backbone import *
from agents.irg.modules.core import BaseLayer, _layer_init

class PolicyEncoder(BaseLayer):
    def __init__(self, 
            obs_inchannel:int = 4, obs_outchannel:int = 64, act_inchannel: int = 1, 
            backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.obs_encoder = self.backbone_mapping[backbone_index][backbone_scale]["encoder"](inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 128)).to(device)
        self.policy_embed_net = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, action):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, action), dim = 1)
        policy_embed = self.policy_embed_net(concat)
        return policy_embed, obs_feature

class PolicyDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, policy_embedding_size:int = 64, device:str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.mlp_decoder = MLP(
            channels=[obs_encoded_size + policy_embedding_size, 64, 32, 1], 
            device = self.device).to(device)

    def forward(self, obs_encoded, policy_embedding):
        # obs_encoded = self.obs_encoder(obs)
        concat = torch.cat((obs_encoded, policy_embedding), dim = 1)
        return self.mlp_decoder(concat)

class TrajectoryEncoder(BaseLayer):
    def __init__(self, obs_inchannel:int = 4, obs_outchannel:int = 64, act_inchannel:int = 2, 
        backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.obs_encoder = self.backbone_mapping[backbone_index][backbone_scale]["encoder"](inchannel=obs_inchannel, outchannel=obs_outchannel).to(device)
        # self.dense = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 256)).to(device)
        self.trajectory_embed_net = _layer_init(nn.Linear(obs_outchannel+act_inchannel, 64)).to(device)
    
    def forward(self, obs, prev_action, prev_reward):
        obs_feature = self.obs_encoder(obs)
        concat = torch.cat((obs_feature, prev_action, prev_reward), dim = 1)
        trajectory_embed = self.trajectory_embed_net(concat)
        return trajectory_embed, obs_feature

class ObservationDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, trajectory_embedding_size:int = 64, 
        backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.obs_decoder = self.backbone_mapping[backbone_index][backbone_scale]["decoder"](inchannel=obs_encoded_size+trajectory_embedding_size+1).to(device)
    
    def forward(self, obs_encoded, action, trajectory_embedding):
        concat = torch.cat((obs_encoded, action, trajectory_embedding), dim = 1)
        pred_obs = self.obs_decoder(concat)
        return pred_obs

class RewardDecoder(BaseLayer):
    def __init__(self, obs_encoded_size:int = 64, trajectory_embedding_size:int = 64, device:str = "cuda") -> None:
        super().__init__()
        self.device = device
        self.mlp_decoder = MLP(channels=[obs_encoded_size + trajectory_embedding_size + 1, 64, 1], device=self.device).to(device)
    
    def forward(self, obs_encoded, action, trajectory_embedding):
        concat = torch.cat((obs_encoded, action, trajectory_embedding), dim = 1)
        pred_reward = self.mlp_decoder(concat)
        return pred_reward