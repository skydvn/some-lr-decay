import os, sys
sys.path.append(os.getcwd())
import copy
import torch
from torch import nn 
from torch import optim
import pandas as pd
from beautifultable import BeautifulTable
from agents.irg.modules.irg_backbone import *
from tqdm import trange

env_def = {
    "max_cycles" : 124,
    "num_agents" : 4,
    "stack_size" : 4,
    "single_frame_size" : (32, 32)
}

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

class IRG_Brain(nn.Module):
    def __init__(self, backbone_index:int = 4, backbone_scale = "small", device:str = "cuda") -> None:
        super().__init__()
        self.device = device

        self.policy_encoder = PolicyEncoder(obs_inchannel=4, 
                                          obs_outchannel=64, 
                                          act_inchannel=1, 
                                          backbone_index=backbone_index, 
                                          backbone_scale = backbone_scale,
                                          device = self.device).to(self.device)
        
        self.policy_decoder = PolicyDecoder(obs_encoded_size=64, 
                                          policy_embedding_size=64, 
                                          device = self.device).to(self.device)
        
        self.trajectory_encoder = TrajectoryEncoder(obs_inchannel=4, 
                                        obs_outchannel=64, 
                                        act_inchannel=2,  
                                        backbone_index=backbone_index, 
                                        backbone_scale = backbone_scale,
                                        device = self.device).to(self.device)
        
        self.obs_decoder = ObservationDecoder(obs_encoded_size=64, 
                                              trajectory_embedding_size=64,  
                                              backbone_index=backbone_index, 
                                              backbone_scale = backbone_scale,
                                              device = self.device).to(self.device)
        
        self.rew_decoder = RewardDecoder(obs_encoded_size=64, 
                                         trajectory_embedding_size=64,
                                         device = self.device).to(self.device)
    
    def forward(self, curr_obs, 
                    curr_act, 
                    prev_act, 
                    prev_rew,
                    device = "cuda"):
        """ Forward Function

        Args:
            curr_obs (torch.Tensor): Current Observation. Size of [None, stack_size, height, width]
            curr_act (torch.Tensor): Current Action. Size of [None, 1]
            prev_act (torch.Tensor): Previous Action. Size of [None, 1]
            prev_rew (torch.Tensor): Previous Reward. Size of [None, 1]

        Returns:
            tuple: Tuple of predicted observation, predicted action, predicted reward, policy embedding, trajectory embedding
        """
        # policy_embedding, obs_policy_encoded = self.policy_encoder(curr_obs, curr_act)
        # pred_act = self.policy_decoder(obs_policy_encoded.to(device), policy_embedding.to(device))     
        trajectory_embedding, obs_trajectory_encoded = self.trajectory_encoder(curr_obs, prev_act, prev_rew)
        pred_obs = self.obs_decoder(obs_trajectory_encoded.to(device), curr_act, trajectory_embedding.to(device))
        pred_rew = self.rew_decoder(obs_trajectory_encoded.to(device), curr_act, trajectory_embedding.to(device))
        return (pred_obs, pred_rew)


class IRG:
    def __init__(self, batch_size: int = 20, lr: float = 0.005, gamma: float = 0.99,
            optimizer: str = "Adam", agent_name: str = None, epochs:int = 3,
            env_dict = env_def, train_device = "cuda", buffer_device = "cpu",
            merge_loss = True, save_path = None, round_scale = 2,
            backbone_scale = "small", debug_mode = False) -> None:
        """Constuctor of DUME

        Args:
            batch_size (int, optional): _description_. Defaults to 20.
            lr (float, optional): _description_. Defaults to 0.005.
            gamma (float, optional): _description_. Defaults to 0.99.
            optimizer (str, optional): _description_. Defaults to "Adam".
            agent_name (str, optional): _description_. Defaults to None.
            epoches (int, optional): _description_. Defaults to 3.
            env_dict (_type_, optional): _description_. Defaults to env_def.
            device (str, optional): _description_. Defaults to "cpu".
        """
        # overall propertise
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.merge_loss = merge_loss
        self.save_path = save_path
        self.round_scale = round_scale
        self.train_device = train_device
        self.buffer_device = buffer_device
        self.agent_name = agent_name
        self.epochs = epochs
        self.env_dict = env_dict
        self.debug_mode = debug_mode
        self.brain = copy.deepcopy(IRG_Brain(
            device = self.train_device, 
            backbone_index=self.env_dict["num_agents"],
            backbone_scale = backbone_scale
            ).to(device = self.train_device))
        self.optimizer = opt_mapping[optimizer](self.brain.parameters(), lr=self.lr)

        self.tracking_brain = copy.deepcopy(IRG_Brain(
            device = self.train_device, 
            backbone_index=self.env_dict["num_agents"],
            backbone_scale = backbone_scale
            ).to(device = self.train_device))
        self.tracking_brain.load_state_dict(self.brain.state_dict())

        # memory replay
        self.rb_obs = list()
        self.rb_act = list()
        self.rb_rew = list()

        # Track
        self.log = {
            "epoch" : [],
            "tel" : [],
            "pel" : [],
            "rl" : [],
            "ol" : [],
            "al" : [],
        }
    
    def add_memory(self, obs: torch.Tensor, acts: torch.Tensor, rews: torch.Tensor):
        """_summary_

        Args:
            obs (torch.Tensor): Observation or batch of observation. Size of [max_cycle, stack_size, height, width]
            acts (torch.Tensor): Action or batch of action. Size of [max_cycle, 1]
            rews (torch.Tensor): Reward or batch of reward gained. Size of [max_cycle, 1]
        """
        self.rb_obs.append(obs)
        self.rb_act.append(acts)
        self.rb_rew.append(rews)
    
    def fake_memory(self):
        for i in range(3):
            obs_buffer = torch.randn((self.env_dict["max_cycles"], self.env_dict["stack_size"], 
                    *tuple(self.env_dict["single_frame_size"]))).to(self.buffer_device)
            act_buffer = torch.randn((self.env_dict["max_cycles"], 1)).to(self.buffer_device)
            rew_buffer = torch.randn((self.env_dict["max_cycles"], 1)).to(self.buffer_device)

            self.add_memory(obs = obs_buffer, acts = act_buffer, rews=rew_buffer)
    
    def __call__(self, curr_obs, curr_act, prev_act, prev_rew):
        pred_obs, pred_rew = self.brain(
            curr_obs.to(device = self.train_device), 
            curr_act.to(device = self.train_device), 
            prev_act.to(device = self.train_device), 
            prev_rew.to(device = self.train_device),
            self.train_device)
        
        scale_pred_obs = ((pred_obs - pred_obs.min()) * (1/(pred_obs.max() - pred_obs.min()) * 255)).type(torch.int8)

        return scale_pred_obs, pred_rew
    
    def unitest(self):
        test_policy_encoder = self.brain.policy_encoder
        test_policy_decoder = self.brain.policy_decoder
        test_trajectory_encoder = self.brain.trajectory_encoder
        test_obs_decoder = self.brain.obs_decoder
        test_rew_decoder = self.brain.rew_decoder

        curr_obs = torch.randn((self.batch_size, self.env_dict["stack_size"], 
                    self.env_dict["single_frame_size"][0], 
                    self.env_dict["single_frame_size"][1])) \
                        # .view(-1, self.env_dict["stack_size"],
                        #     self.env_dict["single_frame_size"][0],
                        #     self.env_dict["single_frame_size"][1])
        prev_obs = torch.randn((self.batch_size, self.env_dict["stack_size"], 
                    self.env_dict["single_frame_size"][0], 
                    self.env_dict["single_frame_size"][1]))\
                        # .view(-1, self.env_dict["stack_size"],
                        #     self.env_dict["single_frame_size"][0],
                        #     self.env_dict["single_frame_size"][1])
        next_obs = torch.randn((self.batch_size, self.env_dict["stack_size"], 
                    self.env_dict["single_frame_size"][0], 
                    self.env_dict["single_frame_size"][1]))\
                        # .view(-1, self.env_dict["stack_size"],
                        #     self.env_dict["single_frame_size"][0],
                        #     self.env_dict["single_frame_size"][1])

        prev_act = torch.randn((self.batch_size, 1))\
            # .view(-1, 1)
        curr_act = torch.randn((self.batch_size, 1))\
            # .view(-1, 1)
        next_act = torch.randn((self.batch_size, 1))\
            # .view(-1, 1)

        prev_rew = torch.randn((self.batch_size, 1))\
            # .view(-1, 1)
        curr_rew = torch.randn((self.batch_size, 1))\
            # .view(-1, 1)
        next_rew = torch.randn((self.batch_size, 1))\
            # .view(-1, 1)

        print(f"curr_obs: {curr_obs.shape}")
        print(f"prev_obs: {prev_obs.shape}")
        print(f"next_obs: {next_obs.shape}")
        print(f"prev_act: {prev_act.shape}")
        print(f"curr_act: {curr_act.shape}")
        print(f"next_act: {next_act.shape}")
        print(f"prev_rew: {prev_rew.shape}")
        print(f"curr_rew: {curr_rew.shape}")
        print(f"next_rew: {next_rew.shape}")

        policy_embedding, obs_policy_encoded = test_policy_encoder(curr_obs, curr_act)
        print(f"policy_embedding - obs_policy_encoded: {policy_embedding.shape} - {obs_policy_encoded.shape}")
        pred_act = test_policy_decoder(obs_policy_encoded, policy_embedding)
        print(f"pred_act: {pred_act.shape}")
        trajectory_embedding, obs_trajectory_encoded = test_trajectory_encoder(curr_obs, prev_act, prev_rew)
        print(f"trajectory_embedding - obs_trajectory_encoded: {trajectory_embedding.shape} - {obs_trajectory_encoded.shape}")
        pred_obs = test_obs_decoder(obs_trajectory_encoded, curr_act, trajectory_embedding)
        print(f"pred_obs: {pred_obs.shape}")
        pred_rew = test_rew_decoder(obs_trajectory_encoded, curr_act, trajectory_embedding)
        print(f"pred_rew: {pred_rew.shape}")

        pred_obs, pred_rew = self(curr_obs, prev_act, curr_act, prev_rew)
        print(f"pred obs: {pred_obs.shape}")
        # print(f"pred act: {pred_act.shape}")
        print(f"pred rew: {pred_rew.shape}")
        # print(f"policy_emb: {policy_emb.shape}")
        # print(f"trajectory_emb: {trajectory_emb.shape}")

        self.fake_memory()
        
        print(f"rb_obs: {len(self.rb_obs)} - {self.rb_obs[0].shape}")
        print(f"rb_act: {len(self.rb_act)} - {self.rb_act[0].shape}")
        print(f"rb_rew: {len(self.rb_rew)} - {self.rb_rew[0].shape}")

        z_emb = torch.randn((5, 128))
        prior_z_emb = torch.randn((5, 128))

        trajectory_dis = self.trajectory_latent_distance(z = z_emb, z1= prior_z_emb)

        print(f"trajectory_dis: {trajectory_dis}")

        # trajectory_encoder_loss
        tel = self.trajectory_encoder_loss(curr_obs, prev_obs, next_obs, prev_act, curr_act, prev_rew, curr_rew, next_rew)
        tel.backward(retain_graph=True)
        print(f"trajectory_encoder_loss: {tel} - {tel.shape}")

        # policy_encoder_loss
        pel = self.policy_encoder_loss(prev_obs, prev_act)
        pel.backward(retain_graph=True)
        print(f"policy_encoder_loss: {pel} - {pel.shape}")

        # reward_loss
        rl = self.reward_loss(obs_trajectory_encoded, curr_act, trajectory_embedding, curr_rew)
        rl.backward(retain_graph=True)
        print(f"reward_loss: {rl} - {rl.shape}")

        # observation_loss
        ol = self.observation_loss(obs_trajectory_encoded, curr_act, trajectory_embedding, next_obs)
        ol.backward(retain_graph=True)
        print(f"observation_loss: {ol} - {ol.shape}")

        # action_loss
        al = self.action_loss(obs_policy_encoded, policy_embedding, curr_act)
        al.backward(retain_graph=True)
        print(f"action_loss: {al} - {al.shape}")

        # train
        self.update()
    
    def update(self):
        # Log 
        print(f"DUME Update for agent {self.agent_name}")

        self.brain.train()

        lowest_total_loss = 0
        
        for epoch in trange(self.epochs):
            for episode in range(len(self.rb_obs)):
                for index in range(1, self.rb_obs[episode].shape[0] - self.batch_size - 2):
                    prev_obs = (self.rb_obs[episode][index:index+self.batch_size]/255).to(self.train_device, dtype = torch.float)
                    curr_obs = (self.rb_obs[episode][index+1:index+self.batch_size+1]/255).to(self.train_device, dtype = torch.float)
                    next_obs = (self.rb_obs[episode][index+2:index+self.batch_size+2]/255).to(self.train_device, dtype = torch.float)

                    prev_act = (self.rb_act[episode][index:index+self.batch_size]).to(self.train_device, dtype = torch.float)
                    curr_act = (self.rb_act[episode][index+1:index+self.batch_size+1]).to(self.train_device, dtype = torch.float)
                    # next_act = self.rb_act[episode][index+2:index+self.batch_size+2]

                    prev_rew = (self.rb_rew[episode][index:index+self.batch_size]).to(self.train_device, dtype = torch.float)
                    curr_rew = (self.rb_act[episode][index+1:index+self.batch_size+1]).to(self.train_device, dtype = torch.float)
                    next_rew = (self.rb_act[episode][index+2:index+self.batch_size+2]).to(self.train_device, dtype = torch.float)

                    tel = self.trajectory_encoder_loss(curr_obs, prev_obs, next_obs, prev_act, curr_act, prev_rew, curr_rew, next_rew)

                    pel = self.policy_encoder_loss(prev_obs, prev_act)

                    trajectory_embedding, obs_trajectory_encoded = self.brain.trajectory_encoder(curr_obs, prev_act, prev_rew)

                    rl = self.reward_loss(obs_trajectory_encoded, curr_act, trajectory_embedding, curr_rew)

                    ol = self.observation_loss(obs_trajectory_encoded, curr_act, trajectory_embedding, next_obs)

                    policy_embedding, obs_policy_encoded = self.brain.policy_encoder(curr_obs, curr_act)

                    al = self.action_loss(obs_policy_encoded, policy_embedding, curr_act)

                    total_loss = tel + pel + rl + ol + al

                    self.optimizer.zero_grad()
                    if self.merge_loss:
                        total_loss.backward()
                    else:
                        tel.backward(retain_graph=True)
                        pel.backward(retain_graph=True)
                        rl.backward(retain_graph=True)
                        ol.backward(retain_graph=True)
                        al.backward(retain_graph=True)
                    self.optimizer.step()
            
            if epoch == 0:
                lowest_total_loss = total_loss
            else:
                if total_loss < lowest_total_loss:
                    lowest_total_loss = total_loss
                    self.tracking_brain.load_state_dict(self.brain.state_dict())
            
            self.logging(
                epoch=epoch, 
                tel = tel.item(), 
                pel = pel.item(), 
                rl = rl.item(), 
                ol = ol.item(), 
                al = al.item())
            
            if self.debug_mode:
                base_df = pd.DataFrame(self.log)
                current_epoch_log = base_df[base_df["epoch"] == epoch].to_dict(orient='list')
                debug_table = BeautifulTable(maxwidth=200)
                debug_table.rows.append(["Epoch", str(epoch), "Last Total Loss", total_loss.item(), "Lowest Total Loss", lowest_total_loss.item(), "", ""])
                debug_table.rows.append([
                    "Last Trajectory Encoder", tel.item(), 
                    "Min Trajectory Encoder", min(current_epoch_log["tel"]),
                    "Max Trajectory Encoder", max(current_epoch_log["tel"]),
                    "Mean Trajectory Encoder", sum(current_epoch_log["tel"])/len(current_epoch_log["tel"])
                ])
                debug_table.rows.append([
                    "Last Policy Encoder", pel.item(), 
                    "Min Policy Encoder", min(current_epoch_log["pel"]),
                    "Max Policy Encoder", max(current_epoch_log["pel"]),
                    "Mean Policy Encoder", sum(current_epoch_log["pel"])/len(current_epoch_log["pel"])
                ])
                debug_table.rows.append([
                    "Last Reward Decoder", rl.item(), 
                    "Min Reward Decoder", min(current_epoch_log["rl"]),
                    "Max Reward Decoder", max(current_epoch_log["rl"]),
                    "Mean Reward Decoder", sum(current_epoch_log["rl"])/len(current_epoch_log["rl"])
                ])
                debug_table.rows.append([
                    "Last Observation Decoder", ol.item(), 
                    "Min Observation Decoder", min(current_epoch_log["ol"]),
                    "Max Observation Decoder", max(current_epoch_log["ol"]),
                    "Mean Observation Decoder", sum(current_epoch_log["ol"])/len(current_epoch_log["ol"])
                ])
                debug_table.rows.append([
                    "Last Action Decoder", al.item(), 
                    "Min Action Decoder", min(current_epoch_log["al"]),
                    "Max Action Decoder", max(current_epoch_log["al"]),
                    "Mean Action Decoder", sum(current_epoch_log["al"])/len(current_epoch_log["al"])
                ])
                print(debug_table)
        try:
            self.model_export(self.save_path, lowest_total_loss, total_loss)
        except Exception as e:
            print("Cannot save model, see the log below")
            print(e)

    def logging(self, epoch, tel, pel, rl, ol, al):
        self.log["epoch"].append(epoch)
        self.log["tel"].append(tel)
        self.log["pel"].append(pel)
        self.log["rl"].append(rl)
        self.log["ol"].append(ol)
        self.log["al"].append(al)
    
    def export_log(self, rdir: str, ep: int, extension: str = ".parquet"):
        """Export log to file
        Args:
            rdir (str): folder for saving
            ep (int): current episode
            extension (str, optional): save file extension. Defaults to ".parquet".
        """
        sub_dir = rdir + "/irg"
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
    
    def model_export(self, rdir: str, lowest_loss, total_loss):
        """Export model to file
        Args:
            dir (str): folder for saving model weights
        """
        sub_dir = rdir + "/irg"
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)    
        agent_sub_dir = sub_dir + f"/{self.agent_name}"
        if not os.path.exists(agent_sub_dir):
            os.mkdir(agent_sub_dir)
        
        filename_last = f"irg_last_{self.agent_name}_{round(total_loss.item(), self.round_scale)}"
        filepath_last = agent_sub_dir + f"/{filename_last}.pt"
        torch.save(self.brain.state_dict(), filepath_last)

        filename_best = f"irg_best_{self.agent_name}_{round(lowest_loss.item(), self.round_scale)}"
        filepath_best = agent_sub_dir + f"/{filename_best}.pt"
        torch.save(self.tracking_brain.state_dict(), filepath_best)

    def trajectory_latent_distance(self, z:torch.Tensor, z1:torch.Tensor = None, reg_weight: float=100) -> torch.Tensor:
        batch_size = z.shape[0]
        reg_weight = reg_weight / (batch_size * (batch_size - 1))
        if z1 is None:
            prior_z = torch.randn(z.shape)
        else:
            prior_z = z1
        
        prior_z__kernel = self.trajectory_latent_kernel(prior_z, prior_z)
        z__kernel = self.trajectory_latent_kernel(z, z)
        priorz_z__kernel = self.trajectory_latent_kernel(prior_z, z)

        mmd = reg_weight * torch.mean(prior_z__kernel) + \
            reg_weight * torch.mean(z__kernel) - \
            2 * reg_weight * torch.mean(priorz_z__kernel)
        return mmd

    def trajectory_latent_kernel(self, x1: torch.Tensor, x2: torch.Tensor = None, eps: float = 1e-7, latent_var: float = 2.) -> torch.Tensor:
        x1d = x1.shape
        x2d = x2.shape

        x1 = x1.view(x1d[0], -1, x1d[1]).to(device=self.train_device)
        x2 = x2.view(-1, x2d[0], x2d[1]).to(device=self.train_device)

        x1 = torch.repeat_interleave(x1, x1d[0], dim=-2)
        x2 = torch.repeat_interleave(x2, x1d[0], dim=-3)

        z_dim = x2.shape[-1]
        C = torch.tensor(2 * z_dim * latent_var).to(self.train_device)
        kernel = C / (torch.tensor(eps).to(self.train_device) + C + torch.square(x1 - x2).sum(dim=-1))

        result = torch.sum(kernel) - torch.trace(kernel)

        return result
    
    def trajectory_encoder_loss(self, curr_obs, prev_obs, next_obs, prev_act, curr_act, prev_rew, curr_rew, next_rew) -> torch.Tensor:
        trajectory_embedding, obs_trajectory_encoded = self.brain.trajectory_encoder(curr_obs, prev_act, prev_rew)
        policy_embedding, obs_policy_encoded = self.brain.policy_encoder(prev_obs, prev_act)

        pred_obs = self.brain.obs_decoder(obs_trajectory_encoded, curr_act, trajectory_embedding)
        pred_rew = self.brain.rew_decoder(obs_trajectory_encoded, curr_act, trajectory_embedding)

        pred_obs = torch.flatten(pred_obs, start_dim=1, end_dim=-1)
        next_obs = torch.flatten(next_obs, start_dim=1, end_dim=-1)

        reconstruction_loss = torch.mean(torch.sum(torch.square(pred_obs - next_obs), dim=1), dim=0)\
                              + torch.mean(torch.sum(torch.square(pred_rew - curr_rew), dim=1), dim=0)

        reg_loss = self.trajectory_latent_distance(trajectory_embedding)
        l2_reg = torch.tensor(0.).to(self.train_device)
        for param in self.brain.trajectory_encoder.parameters():
            l2_reg += torch.norm(param).to(self.train_device)
        
        curr_rew = curr_rew.permute(-1, 0)
        trajectory_policy_dis = torch.sum(torch.square(trajectory_embedding - policy_embedding), dim=-1).view(-1, self.batch_size).permute(-1, 0)
        
        policy_embedding_loss = torch.matmul(curr_rew, trajectory_policy_dis)
        loss = reconstruction_loss + reg_loss + policy_embedding_loss*1e-2 + 1e-3 * l2_reg
        return loss
    
    def policy_encoder_loss(self, prev_obs, prev_act) -> torch.Tensor:
        policy_embedding, obs_policy_encoded = self.brain.policy_encoder(prev_obs, prev_act)
        pred_act = self.brain.policy_decoder(obs_policy_encoded, policy_embedding)

        reconstruction_loss = torch.mean(torch.sum(torch.square(pred_act - prev_act), dim=1), dim=0)
        l2_reg = torch.tensor(0.).to(self.train_device)
        for param in self.brain.policy_encoder.parameters():
            l2_reg += torch.norm(param).to(self.train_device)
        reg_loss = self.trajectory_latent_distance(policy_embedding)

        loss = reconstruction_loss + reg_loss + l2_reg * 1e-3
        return loss

    def reward_loss(self, obs_trajectory_encoded, curr_act, trajectory_embedding, curr_rew) -> torch.Tensor:
        pred_rewards = self.brain.rew_decoder(obs_trajectory_encoded, curr_act, trajectory_embedding)
        l2_reg = torch.tensor(0.).to(self.train_device)
        for param in self.brain.rew_decoder.parameters():
            l2_reg += torch.norm(param).to(self.train_device)
        loss = torch.mean(torch.sum(torch.square(pred_rewards - curr_rew), dim=1), dim=0) + 1e-3 * l2_reg
        return loss

    def observation_loss(self, obs_trajectory_encoded, curr_act, trajectory_embedding, next_obs) -> torch.Tensor:
        pred_obs = self.brain.obs_decoder(obs_trajectory_encoded, curr_act, trajectory_embedding)
        l2_reg = torch.tensor(0.).to(self.train_device)
        for param in self.brain.obs_decoder.parameters():
            l2_reg += torch.norm(param).to(self.train_device)
        
        pred_obs = torch.flatten(pred_obs, start_dim=1, end_dim=-1)
        next_obs = torch.flatten(next_obs, start_dim=1, end_dim=-1)

        loss = torch.mean(torch.sum(torch.square(pred_obs - next_obs), axis=1), axis=0) + 1e-3 * l2_reg
        return loss

    def action_loss(self, obs_policy_encoded, policy_embedding, curr_act) -> torch.Tensor:
        pred_act = self.brain.policy_decoder(obs_policy_encoded, policy_embedding)
        l2_reg = torch.tensor(0.).to(self.train_device)
        for param in self.brain.policy_decoder.parameters():
            l2_reg += torch.norm(param).to(self.train_device)
        loss = torch.mean(torch.sum(torch.square(pred_act - curr_act), axis=1), axis=0) + 1e-3 * l2_reg
        return loss

if __name__ == "__main__":
    env_test = {
        "max_cycles" : 124,
        "num_agents" : 4,
        "stack_size" : 4,
        "single_frame_size" : (32, 32)
    }
    dume = IRG(epochs = 3, 
               batch_size=20, 
               env_dict = env_test, 
               train_device="cpu", 
               buffer_device="cpu", 
               backbone_scale = "small", 
               debug_mode=True)
    dume.unitest() 