from functools import partial

import numpy as np

import rlgym_sim
from rlgym_sim.utils import common_values
from rlgym_sim.utils.action_parsers import ContinuousAction
from rlgym_sim.utils.obs_builders import DefaultObs

from rlrl.consts import TICK_SKIP, TEAM_SIZE
from rlrl.rewards.logger import LoggingCombinedReward

def _build_env(rewards, terminal_conditions, state_setter):
    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
    )

    env = rlgym_sim.make(tick_skip=TICK_SKIP,
                         team_size=TEAM_SIZE,
                         spawn_opponents=True,
                         terminal_conditions=terminal_conditions,
                         reward_fn=LoggingCombinedReward(rewards),
                         obs_builder=obs_builder,
                         action_parser=ContinuousAction(),
                         state_setter=state_setter
    )

    return env

def get_env_builder(config):
    return partial(_build_env, config["rewards"], config["terminal_conditions"], config["state_setter"])