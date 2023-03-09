import os, sys
sys.path.append(os.getcwd())
import argparse
from agents.ppo.modules.buffer import RolloutBuffer, PPORolloutBuffer
from agents.ppo.modules.backbone import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import torch.distributed as dist
from random import uniform, choice
from torch import optim
import pandas as pd

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

backbone_mapping = {
    "siamese" : ActorCriticSiamese,
    "siamese-small" : ActorCriticSiameseSmall,
    "siamese-nano" : ActorCriticSiameseNano,
    "multi-head" : ActorCriticMultiHead,
    "multi-head-small" : ActorCriticMultiHeadSmall
}

def divide(data: list, chunk_size):
    lst = []
    for i in range(0, len(data), chunk_size):
        lst.append(data[i:i + chunk_size]) 
    return lst

def batch_split(data: list, chunk_size):
    batch_data = divide(data, chunk_size)

    prev_sub_data = None
    for idx, sub_data in enumerate(batch_data):
        if len(sub_data) > 1:
            batch_data[idx] = torch.squeeze(torch.stack(sub_data, dim=0))
        else: 
            batch_data[idx] = torch.squeeze(torch.stack(prev_sub_data, dim=0))
        prev_sub_data = sub_data

    return batch_data

class PPO:
    def __init__(self, 
                 stack_size:int = 4, 
                 action_dim:int = 6, 
                 lr_actor:float = 0.05, 
                 lr_critic:float = 0.05, 
                 gamma:float = 0.99, 
                 K_epochs:int = 2, 
                 eps_clip:float = 0.2, 
                 device:str = "cuda", 
                 optimizer:str = "Adam", 
                 batch_size:int = 16, 
                 agent_name: str = None, 
                 backbone:str = "siamese", 
                 debug_mode = None,
                 exp_mem_replay = False,
                 exp_mem_cap = 5,
                 distributed_buffer = False,
                 distributed_learning = False,
                 distributed_optimizer = False,
                 lr_decay = True,
                 lr_decay_mode = 2,
                 lr_low = float(1e-12)):
        """Constructor of PPO

        Args:
            stack_size (int, optional): size of frames stack. Defaults to 4.
            action_dim (int, optional): number of possible action. Defaults to 6.
            lr_actor (float, optional): actor learning rate. Defaults to 0.0003.
            lr_critic (float, optional): critic learning rate. Defaults to 0.001.
            gamma (float, optional): discount factor. Defaults to 0.99.
            K_epochs (int, optional): number of self training epoch. Defaults to 2.
            eps_clip (float, optional): _description_. Defaults to 0.2.
            device (str, optional): type of device used for training. Defaults to "cpu".
            optimizer (str, optional): type of optimizer. Defaults to "Adam".
            batch_size (int, optional): Batch size for each training step. Defaults to 16.
            backbone (str, optional): Type of backbone using. Defaults to "siamese"
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.stack_size = stack_size
        self.batch_size = batch_size
        self.agent_name = agent_name
        self.debug_mode = debug_mode
        self.exp_mem_replay = exp_mem_replay
        self.distributed_buffer = distributed_buffer
        self.distributed_learning = distributed_learning
        self.distributed_optimizer = distributed_optimizer
        self.lr_critic = lr_critic 
        self.lr_actor = lr_actor
        self.lr_decay = lr_decay
        self.lr_decay_mode = lr_decay_mode
        self.lr_low = lr_low
        self.lr_diff_critic = self.lr_critic - self.lr_low
        self.lr_diff_actor = self.lr_actor - self.lr_low
        self.device = device
        self.opt = optimizer
        self.max_curr_step = 0
        
        if self.exp_mem_replay:
            self.buffer = PPORolloutBuffer(device = self.device, capacity = exp_mem_cap)
        else:
            self.buffer = RolloutBuffer()

        self.policy = backbone_mapping[backbone](stack_size = self.stack_size, num_actions = action_dim).to(device)

        self.actor_opt = opt_mapping[self.opt](self.policy.actor.parameters(), lr = lr_actor)
        self.critic_opt = opt_mapping[self.opt](self.policy.critic.parameters(), lr = self.lr_critic)

        self.policy_old = backbone_mapping[backbone](stack_size = self.stack_size, num_actions = action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def log_init(self):
        return {
            "epoch" : [],
            "actor_loss" : [],
            "critic_loss" : []
        }
    
    def select_action(self, obs: torch.Tensor):
        raise DeprecationWarning
        """choose action from the policy
        this function automatically save observation, action predict and value of policy to the buffer
        the insert_buffer function has to be call after this function to avoid mismatch in buffer.

        Args:
            obs (torch.Tensor): observation of the current agent which has size of [None, stack_size, height, width]

        Returns:
            int: action
        """
        with torch.no_grad():
            action, action_logprob, obs_val = self.policy.act(obs.to(self.device, dtype=torch.float))
            
        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.obs_values.append(obs_val)

        return action.item()
    
    def make_action(self, obs: torch.Tensor):
        with torch.no_grad():
            action, action_logprob, obs_val = self.policy.act(obs.to(self.device, dtype=torch.float))
        return action, action_logprob, obs_val

    def insert_buffer(self, obs: torch.Tensor, 
               act: torch.Tensor, 
               log_probs: torch.Tensor,
               rew: torch.Tensor,
               obs_val: torch.Tensor,
               term: bool):
        
        if self.exp_mem_replay:
            self.buffer.append(obs, 
               act, 
               log_probs,
               rew,
               obs_val,
               term)
        else:
            self.buffer.observations.append(obs)
            self.buffer.actions.append(act)
            self.buffer.logprobs.append(log_probs)
            self.buffer.obs_values.append(obs_val)
            self.buffer.rewards.append(rew)
            self.buffer.is_terminals.append(term)
    
    def buffer_sample(self, sample_count = 125, obs_size = (64, 64)):
        raise DeprecationWarning
        """Sampling a random buffer for testing train function

        Args:
            sample_count (int, optional): Number of sample which will be added to the buffer. Defaults to 100.
            obs_size (tuple, optional): frame_size of the env. Defaults to (64, 64).
        """
        for _ in range(sample_count):
            random_obs = torch.randn((self.stack_size, obs_size[0], obs_size[1])).view(-1, 4, 64, 64)
            random_act = self.select_action(random_obs)

            self.buffer.rewards.append(uniform(-1, 1))
            self.buffer.is_terminals.append(choice([True, False]))
        
        old_obs = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_obs_values = torch.squeeze(torch.stack(self.buffer.obs_values, dim=0)).detach().to(self.device)

        print("old_obs: {}".format(old_obs.shape))
        print("old_actions: {}".format(old_actions.shape))
        print("old_logprobs: {}".format(old_logprobs.shape))
        print("old_obs_values: {}".format(old_obs_values.shape))
        print("old_rewards: {}".format(len(self.buffer.rewards)))
        print("old_is_terminals: {}".format(len(self.buffer.is_terminals)))
        
    def update(self):
        if self.exp_mem_replay:
            if self.distributed_buffer:
                self._distributed_update()
            else:
                self._non_distributed_update()
        else:
            self._simple_update()
    
    def _non_distributed_update(self):
        self._show_info()

        # Tracking
        self.log = self.log_init()

        for epoch in range(self.K_epochs):

            for ep in range(len(self.buffer)):

                obs_full, act_full, logprobs_full, rew_full, obs_val_full, is_term = self.buffer[ep]

                # reward normalization

                rew_norm = None
                discount_rew = torch.tensor([0]).to(device=self.device)
                for _rew, _term in zip(rew_full, is_term):
                    if _term:
                        discount_rew = torch.tensor([0]).to(device=self.device)
                    
                    discount_rew = _rew + (self.gamma * discount_rew)
                    if rew_norm == None:
                        rew_norm = discount_rew
                    else:
                        rew_norm = torch.cat((rew_norm, discount_rew))

                for batch_idx in range(0, obs_full.shape[0], self.batch_size):

                    try:
                        base_obs = obs_full[batch_idx:batch_idx + self.batch_size]
                        base_act = act_full[batch_idx:batch_idx + self.batch_size]
                        base_logprobs = logprobs_full[batch_idx:batch_idx + self.batch_size]
                        base_obs_val = obs_val_full[batch_idx:batch_idx + self.batch_size]
                        base_rew = rew_norm[batch_idx:batch_idx + self.batch_size]
                    except:
                        if obs_full.shape[0] - batch_idx >= 1:
                            base_obs = obs_full[batch_idx : ]
                            base_act = act_full[batch_idx : ]
                            base_logprobs = logprobs_full[batch_idx : ]
                            base_obs_val = obs_val_full[batch_idx : ]
                            base_rew = rew_norm[batch_idx : ]

                    # cal advantage
                    advantages = base_rew - base_obs_val

                    # Evaluation
                    action_probs = self.policy.actor(base_obs/255)
                    dist = Categorical(logits=action_probs)

                    logprobs = dist.log_prob(base_act)
                    dist_entropy = dist.entropy()
                    obs_values = self.policy.critic(base_obs/255)

                    # match obs_values tensor dimensions with rewards tensor
                    obs_values = torch.squeeze(obs_values)
                    
                    # Finding the ratio (pi_theta / pi_theta__old)
                    ratios = torch.exp(logprobs - base_logprobs)

                    # Finding Surrogate Loss   
                    obj = ratios * advantages
                    obj_clip = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

                    # final loss of clipped objective PPO
                    actor_loss = -torch.min(obj, obj_clip).mean() - 0.01 * dist_entropy.mean()

                    critic_loss = 0.5 * nn.MSELoss()(obs_values, base_rew)

                    # Logging
                    self.logging(
                        epoch=epoch, 
                        actor_loss = actor_loss.item(), 
                        critic_loss = critic_loss.item()
                    )
                            
                    # take gradient step
                    self.actor_opt.zero_grad()
                    actor_loss.backward()
                    self.actor_opt.step()

                    self.critic_opt.zero_grad()
                    critic_loss.backward()
                    self.critic_opt.step()

                    self.policy_old.load_state_dict(self.policy.state_dict())   

        if self.debug_mode == None:
            pass
        elif self.debug_mode == 0:
            print(f"\nActor: {actor_loss.item()} - Critic: {critic_loss.item()}")
        elif self.debug_mode == 1:
            print(f"\nActor: {actor_loss.item()} - Critic: {critic_loss.item()}")

            if str(self.policy.state_dict()) == str(self.policy_old.state_dict()):
                print("Policy is updated")
            if str(self.policy.actor.state_dict()) == str(self.policy_old.actor.state_dict()):
                print("Actor is updated")
            if str(self.policy.critic.state_dict()) == str(self.policy_old.critic.state_dict()):
                print("Critic is updated")
        elif self.debug_mode == 2:
            print(f"Total step: {len(self.buffer.observations)}")
            print("Actor Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["actor_loss"]), max(self.log["actor_loss"]), sum(self.log["actor_loss"])/len(self.log["actor_loss"])
            ))
            print("Critic Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["critic_loss"]), max(self.log["critic_loss"]), sum(self.log["critic_loss"])/len(self.log["critic_loss"])
            ))              
    
    def _distributed_update(self):
        raise NotImplementedError
    
    def _simple_update(self):
        # Log
        self._show_info()

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = [torch.tensor(np.float32(reward)) for reward in rewards]

        # Batch Split
        obs_batch = batch_split(self.buffer.observations, self.batch_size)
        act_batch = batch_split(self.buffer.actions, self.batch_size)
        logprobs_batch = batch_split(self.buffer.logprobs, self.batch_size)
        obs_values_batch = batch_split(self.buffer.obs_values, self.batch_size)
        reward_batch = batch_split(rewards, self.batch_size)

        # Tracking
        self.log = self.log_init()

        # Optimize policy for K epochs
        for e in range(self.K_epochs):

            # Loop through batch
            for idx in range(len(obs_batch)):

                # cal advantage
                advantages = reward_batch[idx].to(self.device) - obs_values_batch[idx].to(self.device)

                # Evaluation
                action_probs = self.policy.actor(obs_batch[idx].to(self.device)/255)
                dist = Categorical(logits=action_probs)

                logprobs = dist.log_prob(act_batch[idx].to(self.device))
                dist_entropy = dist.entropy()
                obs_values = self.policy.critic(obs_batch[idx].to(self.device)/255)

                # match obs_values tensor dimensions with rewards tensor
                obs_values = torch.squeeze(obs_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - logprobs_batch[idx].to(self.device))

                # Finding Surrogate Loss   
                obj = ratios * advantages
                obj_clip = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                actor_loss = -torch.min(obj, obj_clip).mean() - 0.01 * dist_entropy.mean()

                critic_loss = 0.5 * nn.MSELoss()(obs_values, reward_batch[idx].to(self.device))

                # Logging
                self.logging(
                    epoch=e, 
                    actor_loss = actor_loss.item(), 
                    critic_loss = critic_loss.item()
                )
                        
                # take gradient step
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                self.policy_old.load_state_dict(self.policy.state_dict())
            
        if self.debug_mode == None:
            pass
        elif self.debug_mode == 0:
            print(f"\nActor: {actor_loss.item()} - Critic: {critic_loss.item()}")
        elif self.debug_mode == 1:
            print(f"\nActor: {actor_loss.item()} - Critic: {critic_loss.item()}")

            if str(self.policy.state_dict()) == str(self.policy_old.state_dict()):
                print("Policy is updated")
            if str(self.policy.actor.state_dict()) == str(self.policy_old.actor.state_dict()):
                print("Actor is updated")
            if str(self.policy.critic.state_dict()) == str(self.policy_old.critic.state_dict()):
                print("Critic is updated")
        elif self.debug_mode == 2:
            print(f"Total step: {len(self.buffer.observations)}")
            print("Actor Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["actor_loss"]), max(self.log["actor_loss"]), sum(self.log["actor_loss"])/len(self.log["actor_loss"])
            ))
            print("Critic Loss: min -> {0} | max -> {1} | avg -> {2}".format(
                min(self.log["critic_loss"]), max(self.log["critic_loss"]), sum(self.log["critic_loss"])/len(self.log["critic_loss"])
            ))


        # clear buffer
        self.buffer.clear()

    def update_lr(self, end_no_tstep, max_time_step):
        self.max_curr_step += end_no_tstep

        if self.lr_decay_mode == 2:
            self._update_lr_critic(max_time_step=max_time_step)
            self._udpate_lr_actor(max_time_step=max_time_step)
        elif self.lr_decay_mode == 1:
            self._udpate_lr_actor(max_time_step=max_time_step)
        elif self.lr_decay_mode == 0:
            self._update_lr_critic(max_time_step=max_time_step)
        else:
            raise Exception(f"arg lr_decay_mode specifies which learning rate is reduced over training. \
                            It should be 0, 1, or 2 but found {self.lr_decay_mode} instead")
    
    def _update_lr_critic(self, max_time_step):
        self.lr_critic = (1-self.max_curr_step/(max_time_step))*self.lr_diff_critic + self.lr_low
        new_optim_critic = opt_mapping[self.opt](self.policy.critic.parameters(), lr = self.lr_critic)
        new_optim_critic.load_state_dict(self.critic_opt.state_dict())
        self.critic_opt = new_optim_critic
    
    def _udpate_lr_actor(self, max_time_step):
        self.lr_actor = (1-self.max_curr_step/(max_time_step))*self.lr_diff_actor + self.lr_low
        new_optim_actor = opt_mapping[self.opt](self.policy.actor.parameters(), lr = self.lr_actor)
        new_optim_actor.load_state_dict(self.actor_opt.state_dict())
        self.critic_opt = new_optim_actor
    
    def get_critic_lr(self):
        return self.lr_critic
    
    def get_actor_lr(self):
        return self.lr_actor

    def logging(self, epoch = None, actor_loss = None, critic_loss = None):
        self.log["epoch"].append(epoch)
        self.log["actor_loss"].append(actor_loss)
        self.log["critic_loss"].append(critic_loss)

    def export_log(self, rdir: str, ep: int, extension: str = ".parquet"):
        """Export log to file

        Args:
            rdir (str): folder for saving
            ep (int): current episode
            extension (str, optional): save file extension. Defaults to ".parquet".
        """
        sub_dir = rdir + "/ppo"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)    
        agent_sub_dir = sub_dir + f"/{self.agent_name}"
        if not os.path.exists(agent_sub_dir):
            os.mkdir(agent_sub_dir)

        filepath = agent_sub_dir + f"/{ep}{extension}"
        export_df = pd.DataFrame(self.log)

        if extension == ".parquet":
            export_df.to_parquet(filepath)
        elif extension == ".csv":
            export_df.to_csv(filepath)
        elif extension == ".pickle":
            export_df.to_pickle(filepath)
    
    def model_export(self, rdir: str):
        """Export model to file

        Args:
            dir (str): folder for saving model weights
        """
        sub_dir = rdir + "/ppo"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)    
        agent_sub_dir = sub_dir + f"/{self.agent_name}"
        if not os.path.exists(agent_sub_dir):
            os.mkdir(agent_sub_dir)
        
        filename = f"ppo_{self.agent_name}"
        filpath = agent_sub_dir + f"/{filename}.pt"
        torch.save(self.policy.state_dict(), filpath)

    def _show_info(self):
        print(f"\nPPO Update for agent {self.agent_name}")  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment
    parser.add_argument("--backbone", type=str, choices=[
        "siamese", "siamese-small", "siamese-nano", "multi-head", "multi-head-small"
        ],
        help="Backbone", default="siamese-nano")
    parser.add_argument("--device", type=str, choices=[
        "cuda", "cpu"
        ],
        help="Device", default="cpu")
    parser.add_argument("--epoch", type=int,
        help="epochs", default=50)
    
    args = parser.parse_args()

    ppo = PPO(backbone=args.backbone, device=args.device, K_epochs=args.epoch)