import os, sys
sys.path.append(os.getcwd())
import torch
from torch.utils.data import Dataset

from spds.torchtensorlist import TorchTensorList


class Buffer:
    def __init__(self) -> None:
        self.actions = None
        self.observations = None
        self.logprobs = None
        self.rewards = None
        self.obs_values = None
        self.is_terminals = None
    
    def clear(self):
        raise NotImplementedError
    

class RolloutBuffer(Buffer):
    def __init__(self) -> None:
        super().__init__()
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.obs_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obs_values[:]
        del self.is_terminals[:]


class PPORolloutBuffer(Buffer, Dataset):
    def __init__(self, capacity:int = 5,
                 device:torch.device = None) -> None:
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = "cpu"

        self.actions = TorchTensorList(device=self.device)
        self.observations = TorchTensorList(device=self.device)
        self.logprobs = TorchTensorList(device=self.device)
        self.rewards = TorchTensorList(device=self.device)
        self.obs_values = TorchTensorList(device=self.device)
        self.is_terminals = []

        self.count = 0
        self.capacity = capacity    

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return (self.observations[idx], 
                self.actions[idx],
                self.logprobs[idx],
                self.rewards[idx],
                self.obs_values[idx],
                self.is_terminals[idx])
    
    def append(self, 
               obs: torch.Tensor, 
               act: torch.Tensor, 
               log_probs: torch.Tensor,
               rew: torch.Tensor,
               obs_val: torch.Tensor,
               term: bool):
        
        if self.count < self.capacity:
            self.count += 1
            self.observations.append(obs)
            self.actions.append(act)
            self.logprobs.append(log_probs)
            self.rewards.append(rew)
            self.obs_values.append(obs_val)
            self.is_terminals.append(term)
        else:
            self.observations.pop()
            self.actions.pop()
            self.logprobs.pop()
            self.rewards.pop()
            self.obs_values.pop()
            self.is_terminals.pop()

            self.observations.append(obs)
            self.actions.append(act)
            self.logprobs.append(log_probs)
            self.rewards.append(rew)
            self.obs_values.append(obs_val)
            self.is_terminals.append(term)
    
    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obs_values[:]
        del self.is_terminals[:]