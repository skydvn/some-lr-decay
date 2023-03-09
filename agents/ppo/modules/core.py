import os, sys
sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def forward(self):
        raise NotImplementedError

class ACSiamese(ActorCritic, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def act(self, obs: torch.Tensor):
        action_probs = self.actor(obs/255)
        dist = Categorical(logits=action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        obs_val = self.critic(obs/255)

        return action.detach(), action_logprob.detach(), obs_val.detach()
    
    def evaluate(self, obs, action):
        action_probs = self.actor(obs/255)
        dist = Categorical(logits=action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        obs_val = self.critic(obs/255)
        
        return action_logprobs, obs_val, dist_entropy

class ACMultiHead(ActorCritic):
    def __init__(self) -> None:
        super().__init__()

    def act(self, obs: torch.Tensor):
        x = self.network(obs/255)
        action_probs = self.actor(x)
        dist = Categorical(logits=action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        obs_val = self.critic(x)

        return action.detach(), action_logprob.detach(), obs_val.detach()
    
    def evaluate(self, obs, action):
        x = self.network(obs/255)
        action_probs = self.actor(x)
        dist = Categorical(logits=action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        obs_val = self.critic(x)
        
        return action_logprobs, obs_val, dist_entropy