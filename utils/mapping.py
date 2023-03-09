from envs.warlords.warlord_env import *
from envs.pong.pong_env import *
from envs.cooperative_pong.coop_pong_env import *
from torch import optim
from agents.ppo.agent import PPO
from agents.irg.agent import IRG

env_mapping = {
    "warlords" : wardlord_env_build,
    "pong" : pong_env_build,
    "coop-pong" : coop_pong_env_build
}

env_parobs_mapping = {
    "warlords" : wardlord_coordinate_obs,
    "pong" : pong_coordinate_obs,
    "coop-pong" : coop_pong_coordinate_obs
}

env_parobs_merge_mapping = {
    "warlords" : wardlord_partial_obs_merge,
    "pong" : pong_partial_obs_merge,
    "coop-pong" : coop_pong_partial_obs_merge
}

opt_mapping = {
    "SGD" : optim.SGD,
    "Adam" : optim.Adam
}

agent_mapping = {
    "ppo" : PPO,
    "irg" : IRG
}

