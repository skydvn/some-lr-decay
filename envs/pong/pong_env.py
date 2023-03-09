from pettingzoo.atari import pong_v3
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
import argparse
import os
import cv2 as cv
import torch
import numpy as np

# Metadata
stack_size = 4
frame_size = (64, 64)
max_cycles = 125
render_mode = "rgb_array"
parralel = True
color_reduc = True

def pong_coordinate_obs(obs: torch.Tensor, p_size = 1):
    """ Calculate the coordinate observation

    Args:
        obs (torch.Tensor): Full Observation. Size of [None, stack_size, height, width]
        p_size (int): scale size of observation, max(p_size) = 2

    Returns:
        dict[str : torch.Tensor]: dict of coordinate observation which has size of (height/2, width/2) and 
        the position is based on agent position

        # Agent Position
            # first_0 -> right side
            # second_0 -> left side
    """

    obs = torch.transpose(obs, 2, 3)

    mid = int(obs.shape[-1]/2)

    agent_obs = {
        "first_0" : obs[:, :, :mid*p_size, :],
        "second_0" : obs[:, :, obs.shape[-1] - mid*p_size:, :]
    }

    return agent_obs

def pong_partial_obs_merge(obs_merges, 
                                frame_size: tuple = (64, 64),
                                stack_size: int = 4, p_size = 1):
    """return merged observation

    Args:
        obs_merges (dict[str: torch.Tensor]): dictionary of observation
        frame_size (tuple, optional): Frame Size. Defaults to (64, 64).
        stack_size (int, optional): Number of frames stacked. Defaults to 4.

    Returns:
        _type_: _description_
    """
    first_0 = obs_merges["first_0"]

    mid = int(first_0.shape[-1]/2)

    output_obs = torch.zeros((first_0.shape[0], stack_size, frame_size[0], frame_size[1]))

    output_obs[:, :, :mid*p_size, :] = obs_merges["first_0"]
    output_obs[:, :, first_0.shape[-1] - mid*p_size:, :] = obs_merges["second_0"]

    return output_obs.to(torch.uint8)

def test_coordinate_obs(obs: torch.Tensor):
    """ return coordinate observation

    Args:
        obs (torch.Tensor): Full Observation: Size of [None, stack_size, height, width]
    """

    obs_dict = pong_coordinate_obs(obs)

    for agent in obs_dict:
        print(f"{agent} - {obs_dict[agent].shape}")

        agent_obs = obs_dict[agent][0].permute(-1, 1, 0).numpy()
        print(f"agent_obs: {np.unique(agent_obs)} - {agent_obs.dtype}")

        cv.imshow(agent, agent_obs)
        cv.waitKey(0)

    full_obs = pong_partial_obs_merge(obs_dict)
    print("full_obs - {}".format(full_obs.shape))

    full_obs = full_obs[0].permute(-1, 1, 0).numpy()
    print(f"full_obs: {np.unique(full_obs)} - {full_obs.dtype}")

    cv.imshow("Full", full_obs)
    cv.waitKey(0)

def batchify(x, device = 'cpu'):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x

def pong_env_build(stack_size: int = stack_size, frame_size: tuple = frame_size,
                        max_cycles: int = max_cycles, render_mode: str = render_mode,
                        parralel: bool = parralel, color_reduc: bool = color_reduc):
    """Environment Making

    Args:
        stack_size (int, optional): Number of frames stacked. Defaults to stack_size.
        frame_size (tuple, optional): Frame size. Defaults to frame_size.
        max_cycles (int, optional): after max_cycles steps all agents will return done. Defaults to max_cycles.
        render_mode (str, optional): Type of render. Defaults to render_mode.
        parralel (bool, optional): Let env run on parralel or not. Defaults to parralel.
        color_reduc (bool, optional): Reduce the color channel. Defaults to color_reduc.

    Returns:
        _type_: Environment
    """

    if parralel:
        env = pong_v3.parallel_env(render_mode=render_mode, 
                            max_cycles=max_cycles)
    else:
        env = pong_v3.env(render_mode=render_mode, 
                            max_cycles=max_cycles)
    
    if color_reduc:
        env = color_reduction_v0(env)
    
    env = resize_v1(env, frame_size[0], frame_size[1])

    if stack_size > 1:
        env = frame_stack_v1(env, stack_size=stack_size)
    env.reset()

    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacksize", type=int, default=4,
                        help="")
    parser.add_argument("--framesize", type=tuple, default=(64, 64),
                        help="")
    parser.add_argument("--maxcycles", type=int, default=125,
                        help="")
    parser.add_argument("--rendermode", type=str, default='rgb_array', 
                        choices = ["rgb_array", "human"],
                        help="")
    parser.add_argument("--parralel", type=bool, default=True, 
                        choices = [False, True],
                        help="")
    parser.add_argument("--colorreduc", type=bool, default=True, 
                        choices = [False, True],
                        help="")
    
    args = parser.parse_args()

    env = pong_env_build(stack_size = args.stacksize, frame_size = args.framesize,
                        max_cycles = args.maxcycles, render_mode = args.rendermode,
                        parralel = args.parralel, color_reduc= args.colorreduc)
    
    print("=" * 80)
    print("Summary of warlords env metadata:")
    print(f"Stack size: {args.stacksize}")
    print(f"Frame size: {args.framesize}")
    print(f"Max cycles: {args.maxcycles}")
    print(f"Render mode: {args.rendermode}")
    print(f"Parallel env computing: {args.parralel}")
    print(f"Color reduction: {args.parralel}")
    print("=" * 80)
    print(f"Number of possible agents: {len(env.possible_agents)}")
    print(f"Example of agent: {env.possible_agents[0]}")
    print(f"Number of actions: {env.action_space(env.possible_agents[0]).n}")
    print(f"Action Space: {env.action_space(env.possible_agents[0])}")
    print(f"Observation Size: {env.observation_space(env.possible_agents[0]).shape}")
    
    env.reset()

    render_array = env.render()
    cv.imwrite(os.getcwd() + "/envs/pong/render.jpg", render_array)

    actions = {a : env.action_space(a).sample() for a in env.possible_agents}
    print("Action: {}".format(actions))

    agents = env.possible_agents
    for agent in agents:
        for i in range(10):
            actions = {a : env.action_space(a).sample() for a in env.possible_agents}
            observation, reward, termination, truncation, info = env.step(actions)
        obs = observation[agent]
        cv.imwrite(os.getcwd() + f"/envs/pong/obs_{agent}.jpg", obs)
    
    observation = 0
    for i in range(124):
        render_array = env.render()
        actions = {
            'first_0': env.action_space('first_0').sample(), 
            'second_0': env.action_space('first_0').sample()
        }
        observation, reward, termination, truncation, info = env.step(actions)

        break
    
    # Agent Position
    # first_0 -> right
    # second_0 -> left

    # observation = env.reset()    

    # Save test obs
    # for agent in agents:
    #     obs = observation[agent]
    #     cv.imwrite(os.getcwd() + f"/envs/pong/obs_{agent}.jpg", obs)
    
    # Check same obs -> Same observation - Full Observation
    # print(np.unique((observation["first_0"] - observation["second_0"])))

    observation = batchify(observation)

    print("observation: {}".format(observation.shape))

    test_draw = observation[0] #.permute(-1, 1, 0).numpy()

    print("first-obs: {}".format(test_draw.shape))

    cv.imshow("Source", test_draw.numpy())
    cv.waitKey(0)

    observation = observation.permute(0, -1, 1, 2)[0].view(-1, 4, 64, 64)

    print(observation.shape)

    test_coordinate_obs(observation)