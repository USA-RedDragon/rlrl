from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, EventReward, FaceBallReward
from rlgym_sim.utils.state_setters.random_state import RandomState
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, BallTouchedCondition

from rlrl.rewards.aerial import InAirReward
from rlrl.consts import GAME_TICK_RATE, TICK_SKIP

config = {
    "rewards": [
        ("event", EventReward(touch=50), 1),
        ("vel_player_to_ball", VelocityPlayerToBallReward(), 5),
        ("face_ball", FaceBallReward(), 1),
        ("in_air", InAirReward(), 0.15),
    ],
    "terminal_conditions": [
        BallTouchedCondition(),
        NoTouchTimeoutCondition(int(round(20 * GAME_TICK_RATE / TICK_SKIP)))
    ],
    "state_setter": RandomState(ball_rand_speed=False, cars_rand_speed=True, cars_on_ground=False),
    "hyperparameters": {
        "timestep_limit": 10e15,
        "exp_buffer_size": 4_600_000,
        "ts_per_iteration": 48_000,
        "sac_policy_delay": 3,
        "sac_batch_size": 3_000,
        "sac_ent_coef": 0.075,
        "sac_learning_rate": 2.6e-4,
        "sac_learning_starts": 100_000,
        "sac_train_freq": 6,
        "sac_max_updates_per_iter": 20_000,
        "sac_gradient_steps": 1,
        "sac_tau": 0.015,
        "sac_gamma": 0.95,
        "policy_layer_sizes": (256, 256),
        "critic_layer_sizes": (256, 256),
        "save_every_ts": 1_000_000,
    }
}
