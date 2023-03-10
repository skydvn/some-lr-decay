import os
import json
from utils.mapping import *
from utils.fol_struc import run_folder_verify
from utils.batchify import *
import argparse
from datetime import datetime
import torch
import random
import numpy as np
from tqdm import trange
import pandas as pd


class Training:
    def __init__(self, args: argparse.PARSER) -> None:
        
        # setup
        self.args = args
        args_dict = vars(self.args)

        self.current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        # seed setup
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        # autotune setup
        torch.backends.cudnn.benchmark = True

        # device setup
        if torch.cuda.is_available():
            self.train_device = torch.device(type = "cuda", index = args_dict["device_index"])
            if args_dict["buffer_device"] == "cuda":
                self.buffer_device = torch.device(type = "cuda", index = args_dict["device_index"])
            else:
                self.buffer_device = args_dict["buffer_device"]
        else:
            self.train_device = "cpu"
            self.buffer_device = "cpu"
        
        # verify
        run_folder_verify(self.current_time)

        # setting path
        self.save_train_set_path = os.getcwd() + f"/run/train/{self.current_time}/settings/settings.json"

        # agent logging save_dir
        self.log_agent_dir = os.getcwd() + f"/run/train/{self.current_time}/log"
        self.model_agent_dir = os.getcwd() + f"/run/train/{self.current_time}/weights"

        # main experiment logging
        self.main_log_dir = os.getcwd() + f"/run/train/{self.current_time}/log/main"

        # Script
        self.script_filename = self.args.script
        self.script_path = os.getcwd() + f"/script/{self.script_filename}.json"

        with open(self.save_train_set_path, "w") as outfile:
            json.dump(args_dict, outfile)
        

        # Hyperparams Setup
        self.env_name = args_dict["env"]
        self.stack_size = args_dict["stack_size"]
        self.frame_size = args_dict["frame_size"]
        self.parallel = args_dict["parallel"]
        self.color_reduction = args_dict["color_reduction"]
        self.render_mode = args_dict["render_mode"]
        self.max_cycles = args_dict["max_cycles"]

        self.episodes = args_dict["ep"]
        self.gamma = args_dict["gamma"]
        self.p_size = args_dict["view"]
        self.fix_reward = args_dict["fix_reward"]
        self.max_reward = args_dict["max_reward"]
        self.inverse_reward = args_dict["inverse_reward"]

        self.agent_algo = args_dict["agent"]
        self.epochs = args_dict["epochs"]
        self.batch_size = args_dict["bs"]
        self.actor_lr = args_dict["actor_lr"]
        self.critic_lr = args_dict["critic_lr"]
        self.optimizer = args_dict["opt"]
        self.debug_mode = args_dict["debug_mode"]
        self.eps_clip = args_dict["eps_clip"]
        self.exp_mem = args_dict["exp_mem"]
        self.dist_cap = args_dict["dist_cap"]
        self.dist_buff = args_dict["dist_buff"]
        self.dist_learn = args_dict["dist_learn"]
        self.dist_opt = args_dict["dist_opt"]
        self.lr_decay = args_dict["lr_decay"]
        self.lr_decay_mode = args_dict["lr_decay_mode"]
        self.lr_low = args_dict["lr_low"]

        self.irg_in_use = args_dict["irg"]
        self.irg_backbone = args_dict["irg_backbone"]
        self.irg_epochs = args_dict["irg_epochs"]
        self.irg_batch_size = args_dict["irg_bs"]
        self.irg_lr = args_dict["irg_lr"]
        self.irg_optimizer = args_dict["irg_opt"]
        self.irg_merge_loss = args_dict["irg_merge_loss"]
        self.irg_round_scale = args_dict["irg_round_scale"]

        # Environment initialization
        self.output_env = env_mapping[self.env_name](stack_size=self.stack_size, frame_size=tuple(self.frame_size),
                                                     max_cycles=self.max_cycles, render_mode=self.render_mode,
                                                     parralel=self.parallel, color_reduc=self.color_reduction)

        # Agent names initialization
        self.agent_names = self.output_env.possible_agents

        # Actor Critic Initialization
        self.main_algo_agents = {name: agent_mapping[self.agent_algo](
            stack_size = self.stack_size,
            action_dim = self.output_env.action_space(self.output_env.possible_agents[0]).n,
            lr_actor = self.actor_lr,
            lr_critic = self.critic_lr,
            gamma = self.gamma,
            K_epochs = self.epochs,
            eps_clip = self.eps_clip,
            device = self.train_device,
            optimizer = self.optimizer,
            batch_size = self.batch_size,
            agent_name = name,
            debug_mode = self.debug_mode,
            exp_mem_replay = self.exp_mem,
            exp_mem_cap = self.dist_cap,
            distributed_buffer = self.dist_buff,
            distributed_learning = self.dist_learn,
            distributed_optimizer = self.dist_opt,
            lr_decay = self.lr_decay,
            lr_decay_mode = self.lr_decay_mode,
            lr_low = self.lr_low
        ) for name in self.agent_names}

        # Environment Params Dictionary for IRG
        self.env_irg_def = {
            "max_cycles": self.max_cycles,
            "num_agents": len(self.agent_names),
            "stack_size": self.stack_size,
            "single_frame_size": (int(self.frame_size[0] / 2),
                                  int(self.frame_size[1] / 2))
        }

        # IRG initialization
        if self.irg_in_use:
            self.irg_agents = {name: agent_mapping["irg"](
                batch_size = self.irg_batch_size,
                lr = self.irg_lr,
                gamma = self.gamma,
                optimizer = self.irg_optimizer,
                agent_name = name,
                epochs = self.irg_epochs,
                env_dict = self.env_irg_def,
                train_device = self.train_device,
                buffer_device = self.buffer_device,
                merge_loss = self.irg_merge_loss, 
                save_path = self.model_agent_dir,
                backbone_scale = self.irg_backbone,
                round_scale = self.irg_round_scale
            ) for name in self.agent_names}

    def train(self):
        if self.args.train_type == "train-parallel":
            self.train_parallel()
        elif self.args.train_type == "train-irg-only":
            self.train_irg_only(agent_name=self.args.agent_choose)
        elif self.args.train_type == "train-algo-only":
            self.train_algo_only()
        elif self.args.train_type == "experiment-dual":
            self.experiment_dual()
        elif self.args.train_type == "experiment-algo":
            self.experiment_algo()
        elif self.args.train_type == "pong-algo-only":
            self.pong_algo_only(max_reward=self.max_reward)
        elif self.args.train_type == "pong-irg-only":
            self.pong_irg_only(agent_name=self.args.agent_choose)
        elif self.args.train_type == "pong-irg-algo":
            self.pong_irg_algo(max_reward=self.max_reward)
    
    def main_log_init(self):
        base_dict = {
            "ep" : [],
            "step" : []
        }

        for agent in self.agent_names:
            base_dict[agent] = []
        
        return base_dict
    
    def pong_irg_algo(self, max_reward = None):
        if not self.args.script:
            raise Exception("script arg cannot be None in this mode")
        if not self.irg_in_use:
            raise Exception("irg need to be True and included path to conduct experiment mode")
        
        script_dict = json.load(open(self.script_path))

        irg_weight_paths = {
            agent: os.getcwd() + f"/run/train/{script_dict[agent]}/weights/irg_{agent}.pt" for agent in
            self.agent_names
        }

        for agent in self.agent_names:
            if self.irg_in_use:
                self.irg_agents[agent].brain.load_state_dict(
                    torch.load(irg_weight_paths[agent], map_location=self.train_device)
                )
        
        for ep in trange(self.episodes):

            reward_log = self.main_log_init()

            win_log = self.main_log_init()

            reward_win = {
                agent : 0 for agent in self.agent_names
            }

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                curr_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                curr_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}

                for step in range(self.max_cycles):
                    
                    try:
                        curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                            -1,
                            self.stack_size,
                            self.frame_size[0],
                            self.frame_size[1]
                        )
                    except:
                        break

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    for agent in self.agent_names:
                        base_agent_curr_obs = agent_curr_obs[agent]  # get obs specific to agent

                        obs_merges = {
                            agent: base_agent_curr_obs
                        }  # Create obs dict with restricted view

                        for others in self.agent_names:
                            if others != agent:
                                predict_obs, _ = self.irg_agents[others](
                                    curr_obs=agent_curr_obs[others].to(device=self.train_device, dtype=torch.float),
                                    curr_act=curr_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_act=prev_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_rew=prev_rew_buffer[others].to(device=self.train_device, dtype=torch.float)
                                )

                                # add others' predicted obs to obs_dict
                                obs_merges[others] = predict_obs

                        # Merge Observation
                        base_agent_merge_obs = env_parobs_merge_mapping[self.env_name](
                            obs_merges=obs_merges,
                            frame_size=tuple(self.frame_size),
                            stack_size=self.stack_size,
                            p_size=self.p_size
                        )

                        # Make action
                        action = self.main_algo_agents[agent].make_action(
                            base_agent_merge_obs.to(device=self.train_device, dtype=torch.float)
                        )

                        # Update buffer
                        curr_act_buffer[agent] = torch.Tensor([[action]])

                    # Extract action from buffer

                    actions = {
                        agent: int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                    }

                    next_obs, rewards, terms, _, _ = self.output_env.step(actions)  # Update Environment

                    # Update termination
                    terms = {
                        agent : True if rewards[agent] == 1 or rewards[agent] == -1 else False for agent in self.agent_names
                    }

                    # Update win
                    for agent_name in self.agent_names:
                        if rewards[agent_name] == -1:
                            reward_win[agent_name] += 1
                    
                    # Inverse Reward
                    if self.inverse_reward:
                        for agent in self.agent_names:
                            rewards[agent] = 0 - rewards[agent]

                    # Fix Reward
                    if self.fix_reward:                        
                        for agent in self.agent_names:
                            if rewards[agent] == 0:
                                rewards[agent] = max_reward/(step + 1)
                            elif rewards[agent] == -1:
                                rewards[agent] = -10
                            else:
                                rewards[agent] = 10
                    
                    reward_log["ep"].append(ep)
                    reward_log["step"].append(step)
                    for agent in self.agent_names:
                        reward_log[agent].append(rewards[agent])

                    # Re-update buffer
                    prev_act_buffer = curr_act_buffer
                    prev_rew_buffer = curr_rew_buffer
                    curr_rew_buffer = {
                        agent: torch.Tensor([[rewards[agent]]]) for agent in rewards
                    }          

                    # Update buffer for algo actor critic
                    for agent in rewards:
                        self.main_algo_agents[agent].insert_buffer(rewards[agent], terms[agent])
                
                # Update no. win in episode
                win_log["ep"].append(ep)
                win_log["step"].append(step)
                for agent in self.agent_names:
                    win_log[agent].append(reward_win[agent])
                    
            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir=self.log_agent_dir, ep=ep)
                self.main_algo_agents[agent].model_export(rdir=self.model_agent_dir)

            reward_log_path = self.main_log_dir + f"/{ep}_reward_log.parquet"
            reward_log_df = pd.DataFrame(reward_log)
            reward_log_df.to_parquet(reward_log_path)

            win_log_path = self.main_log_dir + f"/{ep}_win_log.parquet"
            win_log_df = pd.DataFrame(win_log)
            win_log_df.to_parquet(win_log_path)

    def pong_irg_only(self, agent_name):
        if self.env_name != "pong":
            raise Exception(f"Env must be pong but found {self.env_name} instead")
        if not self.irg_in_use:
            raise Exception("irg need to be True and included path to conduct experiment mode")

        # Create Agent
        irg_agent = self.irg_agents[agent_name]

        # Buffer Memory

        for _ in trange(self.episodes):

            with torch.no_grad():

                self.output_env.reset(seed=None)

                obs_lst, act_lst, rew_lst = [], [], []

                for step in range(self.max_cycles):

                    actions = {a: self.output_env.action_space(a).sample() for a in self.output_env.possible_agents}

                    next_obs, rewards, terms, truncation, _ = self.output_env.step(actions)

                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    action = torch.tensor([actions[agent_name]])

                    act_lst.append(action)

                    reward = torch.tensor([rewards[agent_name]])

                    rew_lst.append(reward)

                    try:
                        curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                            -1,
                            self.stack_size,
                            self.frame_size[0],
                            self.frame_size[1]
                        )
                    except:
                        break

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    obs_used = agent_curr_obs[agent_name][0]

                    obs_lst.append(obs_used)

                obs_stack = torch.stack(obs_lst)
                act_stack = torch.stack(act_lst)
                rew_stack = torch.stack(rew_lst)

                irg_agent.add_memory(obs=obs_stack.to(device=self.buffer_device), 
                                        acts=act_stack.to(device=self.buffer_device), 
                                        rews=rew_stack.to(device=self.buffer_device))

        # irg training
        irg_agent.update()
        irg_agent.export_log(rdir=self.log_agent_dir, ep="all")
        
    def pong_algo_only(self, max_reward = 100):

        # Train and logging in parallel
        if self.env_name != "pong":
            raise Exception(f"Env must be pong but found {self.env_name} instead")
        
        for ep in range(self.episodes):

            reward_log = self.main_log_init()

            win_log = self.main_log_init()

            reward_win = {
                agent : 0 for agent in self.agent_names
            }

            round_cnt = 0

            if self.exp_mem:
                obs_lst = []
                act_lst = {agent : [] for agent in self.agent_names}
                log_prob_lst = {agent : [] for agent in self.agent_names}
                rew_lst = {agent : [] for agent in self.agent_names}
                obs_val_lst = {agent : [] for agent in self.agent_names}
                term_lst = {agent : [] for agent in self.agent_names}

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                for step in range(self.max_cycles):

                    try:
                        curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                            -1,
                            self.stack_size,
                            self.frame_size[0],
                            self.frame_size[1]
                        )
                    except:
                        break # expcetion for maximum score in env

                    # Make action
                    actions, actions_buffer, log_probs_buffer, obs_values_buffer = {}, {}, {}, {}

                    for agent in self.agent_names:
                        curr_act, curr_logprob, curr_obs_val = self.main_algo_agents[agent].make_action(curr_obs)

                        actions[agent] = curr_act.item()
                        actions_buffer[agent] = curr_act
                        log_probs_buffer[agent] = curr_logprob
                        obs_values_buffer[agent] = curr_obs_val

                    next_obs, rewards, terms, _, _ = self.output_env.step(actions)  # Update Environment

                    # Update termination
                    terms = {
                        agent : True if rewards[agent] == 1 or rewards[agent] == -1 else False for agent in self.agent_names
                    } 

                    # Update round
                    if rewards["first_0"]:
                        round_cnt += 1

                    # Update win                    
                    for agent_name in self.agent_names:
                        if rewards[agent_name] == -1:
                            reward_win[agent_name] += 1

                    # Inverse reward
                    if self.inverse_reward:
                        for agent in self.agent_names:
                            rewards[agent] = 0 - rewards[agent]

                    # Fix reward
                    if self.fix_reward:                        
                        for agent in self.agent_names:
                            if rewards[agent] == 0:
                                rewards[agent] = max_reward/(step + 1)
                            elif rewards[agent] == -1:
                                rewards[agent] = -10
                            else:
                                rewards[agent] = 10
                    
                    reward_log["ep"].append(ep)
                    reward_log["step"].append(step)
                    for agent in self.agent_names:
                        reward_log[agent].append(rewards[agent])
                    
                    if not self.exp_mem:
                        for agent in self.agent_names:
                            self.main_algo_agents[agent].insert_buffer(obs = curr_obs, 
                                                                        act = actions_buffer[agent], 
                                                                        log_probs = log_probs_buffer[agent],
                                                                        rew = rewards[agent],
                                                                        obs_val = obs_values_buffer[agent],
                                                                        term = terms[agent])
                    else:
                        obs_lst.append(curr_obs[0])
                        for agent in self.agent_names:
                            act_lst[agent].append(actions_buffer[agent])
                            log_prob_lst[agent].append(log_probs_buffer[agent])
                            rew_lst[agent].append(torch.tensor(np.float32(rewards[agent])))
                            obs_val_lst[agent].append(obs_values_buffer[agent])
                            term_lst[agent].append(terms[agent])
                
                if self.exp_mem:
                    for agent in self.agent_names:
                        self.main_algo_agents[agent].insert_buffer(obs = torch.stack(obs_lst), 
                                                                    act = torch.stack(act_lst[agent]), 
                                                                    log_probs = torch.stack(log_prob_lst[agent]),
                                                                    rew = torch.stack(rew_lst[agent]),
                                                                    obs_val = torch.stack(obs_val_lst[agent]),
                                                                    term = term_lst[agent])

                # Update no. win in episode
                win_log["ep"].append(ep)
                win_log["step"].append(step)
                for agent in self.agent_names:
                    win_log[agent].append(reward_win[agent])

            print(f"Episode: {ep} | Average: {step/round_cnt} | Total: {step} | Round: {round_cnt}")

            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir=self.log_agent_dir, ep=ep)  # Save main algo log
                self.main_algo_agents[agent].model_export(rdir=self.model_agent_dir)  # Save main algo model

                self.main_algo_agents[agent].update_lr(end_no_tstep = ep,
                                                       max_time_step = self.episodes)
                
                # print(self.main_algo_agents[agent].get_critic_lr())
                # print(self.main_algo_agents[agent].get_actor_lr())
            
            reward_log_path = self.main_log_dir + f"/{ep}_reward_log.parquet"
            reward_log_df = pd.DataFrame(reward_log)
            reward_log_df.to_parquet(reward_log_path)

            win_log_path = self.main_log_dir + f"/{ep}_win_log.parquet"
            win_log_df = pd.DataFrame(win_log)
            win_log_df.to_parquet(win_log_path)

    def experiment_algo(self):

        script_dict = json.load(open(self.script_path))

        algo_date = script_dict["algo"]

        algo_weight_paths = {
            agent: os.getcwd() + f"/run/train/{algo_date}/weights/ppo_{agent}.pt" for agent in self.agent_names
        }

        for agent in self.agent_names:
            self.main_algo_agents[agent].policy_old.load_state_dict(
                torch.load(algo_weight_paths[agent], map_location=self.train_device)
            )

        for ep in trange(self.episodes):

            main_log = self.main_log_init()

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                for step in range(self.max_cycles):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1,
                        self.stack_size,
                        self.frame_size[0],
                        self.frame_size[1]
                    )

                    actions = {
                        agent: self.main_algo_agents[agent].make_action(curr_obs) for agent in self.agent_names
                    }

                    next_obs, rewards, terms, truncation, _ = self.output_env.step(actions)  # Update Environment

                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    main_log["ep"].append(ep)
                    main_log["step"].append(step)
                    for agent in self.agent_names:
                        main_log[agent].append(rewards[agent])

                    if len(list(terms.keys())) <= 1:
                        break

            # Save main log
            main_log_path = self.main_log_dir + f"/{ep}.parquet"
            main_log_df = pd.DataFrame(main_log)
            main_log_df.to_parquet(main_log_path)

    def experiment_dual(self):
        if not self.irg_in_use:
            raise Exception("irg need to be True and included path to conduct experiment mode")

        script_dict = json.load(open(self.script_path))

        algo_date = script_dict["algo"]

        irg_weight_paths = {
            agent: os.getcwd() + f"/run/train/{script_dict[agent]}/weights/irg_{agent}.pt" for agent in
            self.agent_names
        }

        algo_weight_paths = {
            agent: os.getcwd() + f"/run/train/{algo_date}/weights/ppo_{agent}.pt" for agent in self.agent_names
        }

        # Load weight

        for agent in self.agent_names:
            if self.irg_in_use:
                self.irg_agents[agent].brain.load_state_dict(
                    torch.load(irg_weight_paths[agent], map_location=self.train_device)
                )
            self.main_algo_agents[agent].policy_old.load_state_dict(
                torch.load(algo_weight_paths[agent], map_location=self.train_device)
            )

        # Conduct experiment

        for ep in trange(self.episodes):

            main_log = self.main_log_init()

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                curr_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                curr_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}

                for step in range(self.max_cycles):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1,
                        self.stack_size,
                        self.frame_size[0],
                        self.frame_size[1]
                    )

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    for agent in self.agent_names:
                        base_agent_curr_obs = agent_curr_obs[agent]  # get obs specific to agent

                        obs_merges = {
                            agent: base_agent_curr_obs
                        }  # Create obs dict with restricted view

                        for others in self.agent_names:
                            if others != agent:
                                predict_obs, _ = self.irg_agents[others](
                                    curr_obs=agent_curr_obs[others].to(device=self.train_device, dtype=torch.float),
                                    curr_act=curr_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_act=prev_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_rew=prev_rew_buffer[others].to(device=self.train_device, dtype=torch.float)
                                )

                                # add others' predicted obs to obs_dict
                                obs_merges[others] = predict_obs

                        # Merge Observation
                        base_agent_merge_obs = env_parobs_merge_mapping[self.env_name](
                            obs_merges=obs_merges,
                            frame_size=tuple(self.frame_size),
                            stack_size=self.stack_size,
                            p_size=self.p_size
                        )

                        # Make action
                        action = self.main_algo_agents[agent].make_action(
                            base_agent_merge_obs.to(device=self.train_device, dtype=torch.float)
                        )

                        # Update buffer
                        curr_act_buffer[agent] = torch.Tensor([[action]])

                    # Extract action from buffer

                    actions = {
                        agent: int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                    }

                    next_obs, rewards, terms, _, _ = self.output_env.step(actions)  # Update Environment

                    # Fix reward
                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    main_log["ep"].append(ep)
                    main_log["step"].append(step)
                    for agent in self.agent_names:
                        main_log[agent].append(rewards[agent])

                    # Re-update buffer
                    prev_act_buffer = curr_act_buffer
                    prev_rew_buffer = curr_rew_buffer
                    curr_rew_buffer = {
                        agent: torch.Tensor([[rewards[agent]]]) for agent in rewards
                    }

                    if len(list(terms.keys())) <= 1:
                        break

            main_log_path = self.main_log_dir + f"/{ep}.parquet"
            main_log_df = pd.DataFrame(main_log)
            main_log_df.to_parquet(main_log_path)

    def train_irg_only(self, agent_name="first_0"):

        if not self.irg_in_use:
            raise Exception("irg need to be True and included path to conduct experiment mode")

        # Create Agent
        irg_agent = self.irg_agents[agent_name]

        # Buffer Memory

        for _ in trange(self.episodes):

            with torch.no_grad():

                self.output_env.reset(seed=None)

                obs_lst, act_lst, rew_lst = [], [], []

                for step in range(self.max_cycles):

                    actions = {a: self.output_env.action_space(a).sample() for a in self.output_env.possible_agents}

                    next_obs, rewards, terms, truncation, _ = self.output_env.step(actions)

                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    action = torch.tensor([actions[agent_name]])

                    act_lst.append(action)

                    reward = torch.tensor([rewards[agent_name]])

                    rew_lst.append(reward)

                    if len(list(terms.keys())) <= 1:
                        break

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1,
                        self.stack_size,
                        self.frame_size[0],
                        self.frame_size[1]
                    )

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    obs_used = agent_curr_obs[agent_name][0]

                    obs_lst.append(obs_used)

                obs_stack = torch.stack(obs_lst)
                act_stack = torch.stack(act_lst)
                rew_stack = torch.stack(rew_lst)

                irg_agent.add_memory(obs=obs_stack, acts=act_stack, rews=rew_stack)

        # irg training
        irg_agent.update()
        irg_agent.export_log(rdir=self.log_agent_dir, ep="all")
        irg_agent.model_export(rdir=self.model_agent_dir)

    def train_algo_only(self):

        for ep in range(self.episodes):

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                for step in trange(self.max_cycles):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1,
                        self.stack_size,
                        self.frame_size[0],
                        self.frame_size[1]
                    )

                    actions = {
                        agent: self.main_algo_agents[agent].select_action(curr_obs) for agent in self.agent_names
                    }

                    next_obs, rewards, terms, truncation, _ = self.output_env.step(actions)  # Update Environment

                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    if len(list(terms.keys())) <= 1:
                        break   

                    for agent in rewards:
                        self.main_algo_agents[agent].insert_buffer(rewards[agent], True if agent in terms else False)

            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir=self.log_agent_dir, ep=ep)  # Save main algo log
                self.main_algo_agents[agent].model_export(rdir=self.model_agent_dir)  # Save main algo model

    def train_parallel(self):

        # Training
        for ep in trange(self.episodes):

            main_log = self.main_log_init()

            with torch.no_grad():

                next_obs = self.output_env.reset(seed=None)

                # temporary buffer
                curr_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_act_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                curr_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}
                prev_rew_buffer = {agent: torch.zeros(1, 1) for agent in self.agent_names}

                for step in range(self.max_cycles):

                    curr_obs = batchify_obs(next_obs, self.buffer_device)[0].view(
                        -1,
                        self.stack_size,
                        self.frame_size[0],
                        self.frame_size[1]
                    )

                    agent_curr_obs = env_parobs_mapping[self.env_name](curr_obs, p_size=self.p_size)

                    for agent in self.agent_names:
                        base_agent_curr_obs = agent_curr_obs[agent]  # get obs specific to agent

                        obs_merges = {
                            agent: base_agent_curr_obs
                        }  # Create obs dict with restricted view

                        for others in self.agent_names:
                            if others != agent:
                                # Add other agent information to their irg memory
                                self.irg_agents[others].add_memory(
                                    agent_curr_obs[others],
                                    curr_act_buffer[others],
                                    curr_rew_buffer[others])

                                # Get predicted information by irg
                                predict_obs, _ = self.irg_agents[
                                    others](
                                    curr_obs=agent_curr_obs[others].to(device=self.train_device, dtype=torch.float),
                                    curr_act=curr_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_act=prev_act_buffer[others].to(device=self.train_device, dtype=torch.float),
                                    prev_rew=prev_rew_buffer[others].to(device=self.train_device, dtype=torch.float)
                                )

                                # add others' predicted obs to obs_dict
                                obs_merges[others] = predict_obs

                        # Merge Observation
                        base_agent_merge_obs = env_parobs_merge_mapping[self.env_name](
                            obs_merges=obs_merges,
                            frame_size=tuple(self.frame_size),
                            stack_size=self.stack_size,
                            p_size=self.p_size
                        )

                        # Make action
                        action = self.main_algo_agents[agent].select_action(
                            base_agent_merge_obs.to(device=self.train_device, dtype=torch.float)
                        )

                        # Update buffer
                        curr_act_buffer[agent] = torch.Tensor([[action]])

                    # Extract action from buffer

                    actions = {
                        agent: int(curr_act_buffer[agent][0].item()) for agent in curr_act_buffer
                    }

                    next_obs, rewards, terms, truncation, _ = self.output_env.step(actions)  # Update Environment

                    if self.fix_reward:
                        rewards = {
                            agent: step if agent in terms else 0 for agent in self.agent_names
                        }

                    # Logging
                    main_log["ep"].append(ep)
                    main_log["step"].append(step)
                    for agent in self.agent_names:
                        main_log[agent].append(rewards[agent])

                    # Update buffer
                    prev_act_buffer = curr_act_buffer
                    prev_rew_buffer = curr_rew_buffer
                    curr_rew_buffer = {
                        agent: torch.Tensor([[rewards[agent]]]) for agent in rewards
                    }

                    if len(list(rewards.keys())) <= 1:
                        break

                    # Update Main Algo Memory
                    for agent in rewards:
                        self.main_algo_agents[agent].insert_buffer(rewards[agent], terms[agent])

            # Update Main Policy and irg 
            for agent in self.agent_names:
                self.main_algo_agents[agent].update()
                self.main_algo_agents[agent].export_log(rdir=self.log_agent_dir, ep=ep)  # Save main algo log
                self.main_algo_agents[agent].model_export(rdir=self.model_agent_dir)  # Save main algo model
                print("\n")
                self.irg_agents[agent].update()
                self.irg_agents[agent].export_log(rdir=self.log_agent_dir, ep=ep)  # Save irg log
                self.irg_agents[agent].model_export(rdir=self.model_agent_dir)  # Save irg model
                print("\n")

            # Save main log
            main_log_path = self.main_log_dir + f"/{ep}.parquet"
            main_log_df = pd.DataFrame(main_log)
            main_log_df.to_parquet(main_log_path)