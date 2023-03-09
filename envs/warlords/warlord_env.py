from pettingzoo.atari import warlords_v3
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

def box_size(frame_size: tuple, p_size):
    pH = frame_size[0]
    pW  = frame_size[1]
    pHc = pH / 2
    pWc = pW / 2

    # agent_idx_map = {
    #     "first_0" : 0,
    #     "second_0" : 1,
    #     "third_0" : 2,
    #     "fourth_0" : 3
    #     }
    
    # agent_allocator = {
    #     agent : (((agent_idx_map[agent]+2)%4)//2, ((agent_idx_map[agent]+2)%4)%2) for agent in agent_idx_map
    # }

    agent_allocator = {
        "first_0" : (0, 0),
        "second_0" : (1, 0),  
        "third_0" : (0, 1),
        "fourth_0" : (1, 1)
        }

    agent_box_coordinator = {
        agent : (
            pH*agent_allocator[agent][0] + p_size*(pHc - agent_allocator[agent][0]*pH),
            pH*agent_allocator[agent][0],
            pW*agent_allocator[agent][1] + p_size*(pWc - agent_allocator[agent][1]*pW),
            pW*agent_allocator[agent][1]
        ) for agent in agent_allocator
    }

    agent_box_indices = {
        agent : (
            int(min(agent_box_coordinator[agent][0],agent_box_coordinator[agent][1])),
            int(max(agent_box_coordinator[agent][0],agent_box_coordinator[agent][1])),
            int(min(agent_box_coordinator[agent][2],agent_box_coordinator[agent][3])),
            int(max(agent_box_coordinator[agent][2],agent_box_coordinator[agent][3]))
        ) for agent in agent_box_coordinator
    }

    return agent_box_indices

def wardlord_coordinate_obs(obs: torch.Tensor, p_size = 1):
    """ Calculate the coordinate observation

    Args:
        obs (torch.Tensor): Full Observation. Size of [None, stack_size, height, width]
        p_size (int): scale size of observation, max(p_size) = 2

    Returns:
        dict[str : torch.Tensor]: dict of coordinate observation which has size of (height/2, width/2) and 
        the position is based on agent position

        # Agent Position
            # first_0 -> top left
            # second_0 -> top right
            # third_0 -> bottom left
            # fourth_0 -> bottom right
    """

    obs = torch.transpose(obs, 2, 3)

    agent_box_indices = box_size(frame_size=(obs.shape[2], obs.shape[3]), p_size=p_size)

    agent_obs = {
        agent : obs[
            :, :,
            agent_box_indices[agent][0]:agent_box_indices[agent][1],
            agent_box_indices[agent][2]:agent_box_indices[agent][3]
        ] for agent in agent_box_indices
    }

    return agent_obs

def wardlord_partial_obs_merge(obs_merges, 
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

    agent_box_indices = box_size(frame_size=frame_size, p_size=p_size)

    output_obs = torch.zeros((first_0.shape[0], stack_size, frame_size[0], frame_size[1]))

    for agent in agent_box_indices:
        output_obs[
            :, :, 
            agent_box_indices[agent][0]:agent_box_indices[agent][1],
            agent_box_indices[agent][2]:agent_box_indices[agent][3]
        ] = obs_merges[agent]

    return output_obs.to(torch.uint8)

def test_coordinate_obs(obs: torch.Tensor):
    """ return coordinate observation

    Args:
        obs (torch.Tensor): Full Observation: Size of [None, stack_size, height, width]
    """

    obs_dict = wardlord_coordinate_obs(obs)

    for agent in obs_dict:
        print(f"{agent} - {obs_dict[agent].shape}")

        agent_obs = obs_dict[agent][0].permute(-1, 1, 0).numpy()
        print(f"agent_obs: {np.unique(agent_obs)} - {agent_obs.dtype}")

        cv.imshow(agent, agent_obs)
        cv.waitKey(0)

    full_obs = wardlord_partial_obs_merge(obs_dict)
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


# Env setup
def wardlord_env_build(stack_size: int = stack_size, frame_size: tuple = frame_size,
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
    # assert type(stack_size) != int, "stack_size argument should be an integer"
    # assert type(frame_size) != tuple, "frame_size argument should be an tuple or list"
    # assert type(max_cycles) != int, "max_cycles argument should be an integer"
    # assert type(render_mode) != str, "render_mode argument should be an string: rgb_array or human"
    # assert type(parralel) != bool, "parralel argument should be an boolean"
    # assert type(color_reduc) != bool, "color_reduc argument should be an boolean"

    # assert stack_size == None, "stack_size argument should not be None"
    # assert frame_size == None, "frame_size argument should not be None"
    # assert max_cycles == None, "max_cycles argument should not be None"
    # assert render_mode == None, "render_mode argument should not be None"
    # assert parralel == None, "parralel argument should not be None"
    # assert color_reduc == None, "color_reduc argument should not be None"

    if parralel:
        env = warlords_v3.parallel_env(render_mode=render_mode, 
                            max_cycles=max_cycles)
    else:
        env = warlords_v3.env(render_mode=render_mode, 
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

    env = wardlord_env_build(stack_size = args.stacksize, frame_size = args.framesize,
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
    # cv.imwrite(os.getcwd() + "/envs/warlords/render.jpg", render_array)

    actions = {a : env.action_space(a).sample() for a in env.possible_agents}
    print("Action: {}".format(actions))

    agents = env.possible_agents
    # for agent in agents:
    #     for i in range(10):
    #         actions = {a : env.action_space(a).sample() for a in env.possible_agents}
    #         observation, reward, termination, truncation, info = env.step(actions)
    #     obs = observation[agent]
    #     cv.imwrite(os.getcwd() + f"/envs/warlords/obs_{agent}.jpg", obs)

    observation = 0
    for i in range(100):
        actions = {
            'first_0': env.action_space('first_0').sample(), 
            'second_0': env.action_space('first_0').sample(), 
            'third_0': env.action_space('first_0').sample(), 
            'fourth_0': env.action_space('first_0').sample()}
        observation, reward, termination, truncation, info = env.step(actions)
        
    
    # Agent Position
    # first_0 -> top left
    # second_0 -> top right
    # third_0 -> bottom left
    # fourth_0 -> bottom right

    # observation = env.reset()    

    # Save test obs
    # for agent in agents:
    #     obs = observation[agent]
    #     cv.imwrite(os.getcwd() + f"/envs/warlords/obs_{agent}.jpg", obs)
    
    # Check same obs -> Same observation - Full Observation
    # print(np.unique((observation["first_0"] - observation["second_0"])))
    # print(np.unique((observation["first_0"] - observation["third_0"])))
    # print(np.unique((observation["first_0"] - observation["fourth_0"])))

    observation = batchify(observation)

    print("observation: {}".format(observation.shape))

    test_draw = observation[0] #.permute(-1, 1, 0).numpy()

    print("first-obs: {}".format(test_draw.shape))

    cv.imshow("Source", test_draw.numpy())
    cv.waitKey(0)

    observation = observation.permute(0, -1, 1, 2)[0].view(-1, 4, 64, 64)

    print(observation.shape)

    test_coordinate_obs(observation)

    # observation = torch.transpose(observation, 2, 3)
    # print(observation.shape)

    # test_coordinate_obs(observation)