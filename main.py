import os, sys
import argparse
from datetime import datetime
import torch
from beautifultable import BeautifulTable

from manager import Training

if __name__ == '__main__':
    print("__main__")

    parser = argparse.ArgumentParser()
    # Check Mode
    parser.add_argument("--check", type=bool, default=False, choices=[True, False],
                        help="CLI Check")
    # Environment
    parser.add_argument("--env", type=str, default="warlords", choices=["warlords", "pong", "coop-pong"],
                        help="Environment used in training and testing")
    parser.add_argument("--render_mode", type=str, default=None, choices=["rgb_array", "human"],
                        help="Mode of rendering")
    parser.add_argument("--stack_size", type=int, default=4,
                        help="Number of stacking frames")
    parser.add_argument("--max_cycles", type=int, default=124,
                        help="Number of step in one episode")
    parser.add_argument("--frame_size", type=list, default=(64, 64),
                        help="Width and height of frame")
    parser.add_argument("--parallel", type=bool, default=True,
                        help="Process the environment in multi cpu core")
    parser.add_argument("--color_reduction", type=bool, default=True,
                        help="Reduce color to grayscale")
    parser.add_argument("--ep", type=int, default=2,
                        help="Total Episodes")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--view", type=float, default=1,
                        help="Area scale of partial observation varies in range of (0, 2)]")

    # Training
    parser.add_argument("--train_type", type=str, default="train-irg-only",
                        choices=["train-irg-only", "train-parallel", "train-algo-only", "experiment-dual",
                                 "experiment-algo", "pong-algo-only", "pong-irg-only", "pong-irg-algo"],
                        help="Type of training")
    parser.add_argument("--agent_choose", type=str, default="first_0",
                        choices=["first_0", "second_0", "third_0", "fourth_0", "paddle_0", "paddle_1"],
                        help="Agent chose for training, only available for irg or algo irg-only mode")
    parser.add_argument("--script", type=str,
                        help="Script includes weight paths to model, only needed in experiment mode, "
                             "detail in /script folder, create your_setting.json same as sample.json "
                             "for conducting custom experiment")
    parser.add_argument("--fix_reward", type=bool, default=False,
                        help="Make reward by step")
    parser.add_argument("--max_reward", type=int, default=100,
                        help="Max reward only use for pong-algo-only mode")
    parser.add_argument("--inverse_reward", type=bool, default=False,
                        help="change the sign of reward")
    parser.add_argument("--buffer_device", type=str, default="cpu",
                        help="Device used for memory replay")
    parser.add_argument("--device_index", type=int,
                        help="CUDA index used for training")

    # Agent
    parser.add_argument("--agent", type=str, default="ppo", choices=["ppo"],
                        help="Deep policy model architecture")
    parser.add_argument("--backbone", type=str, default="siamese", choices=[
        "siamese", "siamese-small", "siamese-nano", "multi-head", "multi-head-small"],
                        help="PPO Backbone")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epoch for training")
    parser.add_argument("--bs", type=int, default=20,
                        help="Batch size")
    parser.add_argument("--actor_lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--critic_lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--eps_clip", type=float, default=0.2,
                        help="Epsilon clip used in ppo clip gradient")
    parser.add_argument("--opt", type=str, default="Adam",
                        help="Optimizer")
    parser.add_argument("--debug_mode", type=int, default=None, choices=[0, 1, 2],
                        help="Debug mode")
    parser.add_argument("--exp_mem", type=bool, default=False,
                        help="Using experience memory replay")
    parser.add_argument("--dist_buff", type=bool, default=False,
                        help="Using memory distributed experience memory replay")
    parser.add_argument("--dist_cap", type=int, default=5,
                        help="Capacity - Number of episodes stored in dynamic Torch Tensor List")
    parser.add_argument("--dist_learn", type=bool, default=False,
                        help="Learning in multi GPUS")
    parser.add_argument("--dist_opt", type=bool, default=False,
                        help="Gradient Storing multi GPUs")
    parser.add_argument("--lr_decay", type=bool, default=False,
                        help="Learning Rate Scheduler")
    parser.add_argument("--lr_decay_mode", type=int, default=0,
                        help="Learning Rate Decay Modes: 0, 1, 2. They are for \
                            updating learning rate of critic, actor, or both, respectively")
    parser.add_argument("--lr_low", type=float, default=float(1e-12),
                        help="Lowest learning rate achieved")

    # irg
    parser.add_argument("--irg", type=bool, default=True,
                        help="Partial Observation Deep Policy")
    parser.add_argument("--irg_backbone", type=str, default="small", 
                        choices=["small", "normal"],
                        help="Backbone used in training")
    parser.add_argument("--irg_epochs", type=int, default=1,
                        help="Number of epoch for training")
    parser.add_argument("--irg_bs", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--irg_merge_loss", type=bool, default=True,
                        help="Take the gradient in the total loss instead of backwarding each loss separately")
    parser.add_argument("--irg_lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--irg_opt", type=str, default="Adam",
                        help="Optimizer for irg")
    parser.add_argument("--irg_round_scale", type=int, default=2,
                        help="Number of number after comma in decimal")
    args = parser.parse_args()

    table = BeautifulTable(maxwidth=140, detect_numerics = False)
    table.rows.append([args.env, "train_type", args.train_type, "agent", args.agent, "IRG", str(args.irg)])
    table.rows.append([args.stack_size, "agent_choose", args.agent_choose, "backbone", args.backbone, "irg_epochs", args.irg_epochs])
    table.rows.append([args.frame_size, "script", args.script, "epochs", args.epochs, "irg_bs", args.irg_bs])
    table.rows.append([str(args.parallel), "fix_reward", str(args.fix_reward), "bs", args.bs, "irg_lr", args.irg_lr])
    table.rows.append([str(args.color_reduction), "buffer_device", args.buffer_device, "actor_lr", args.actor_lr, "irg_opt", args.irg_opt])
    table.rows.append([args.render_mode, "device_index", args.device_index, "critic_lr", args.critic_lr, "irg_merge_loss", str(args.irg_merge_loss)])
    table.rows.append([args.max_cycles, "", "", "opt", args.opt, "irg_backbone", args.irg_backbone])
    table.rows.append([args.ep, "", "", "eps_clip", args.eps_clip, "irg_round_scale", args.irg_round_scale])
    table.rows.append([args.gamma, "", "", "exp_mem", str(args.exp_mem), "", ""])
    table.rows.append([args.view, "", "", "dist_buff", str(args.dist_buff), "", ""])
    table.rows.append(["", "", "", "dist_cap", args.dist_cap, "", ""])
    table.rows.append(["", "", "", "dist_learn", str(args.dist_learn), "", ""])
    table.rows.append(["", "", "", "dist_opt", str(args.dist_opt), "", ""])
    table.rows.append(["", "", "", "lr_decay", str(args.lr_decay), "", ""])
    table.rows.append(["", "", "", "lr_decay_mode", str(args.lr_decay_mode), "", ""])
    table.rows.append(["", "", "", "lr_row", str(args.lr_low), "", ""])
    table.rows.header = ["env", "stack_size", "frame_size", "parallel", "color_reduc", "render_mode", "max_cycles", "ep", "gamma", "view", "", "", "", "", "", ""]
    table.columns.header = ["ENV INFO", "", "TRAIN INFO", "", "AGENT INFO", "", "IRG INFO"]
    print(table)

    if not torch.cuda.is_available():
        print()
        print("="*10, "CUDA INFO", "="*10)
        print(f"Cuda is not available on this machine")
        print("="*10, "CUDA INFO", "="*10)
        print()
    elif not args.device_index == None:
        if args.device_index > torch.cuda.device_count():
            raise Exception(f"The device chose is higher than the number of available cuda device.\
                There are {torch.cuda.device_count()} but {args.device_index} chose instead")
        else:
            print()
            print("="*10, "CUDA INFO", "="*10)
            print(f"Total number of cuda: {torch.cuda.device_count()}")
            print(f"CUDA current index: {args.device_index}")
            print(f"CUDA device name: {torch.cuda.get_device_name(args.device_index)}")
            print(f"CUDA device address: {torch.cuda.device(args.device_index)}")
            print("="*10, "CUDA INFO", "="*10)
            print()
    else:
        print("="*10, "CUDA INFO", "="*10)
        print("CUDA not in use")
        print("="*10, "CUDA INFO", "="*10)

    if not args.check:
        train = Training(args=args)
        train.train()