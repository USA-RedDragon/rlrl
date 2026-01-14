from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, SaveBoostReward
from rlgym_sim.utils.state_setters.random_state import RandomState
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition

from rlrl.rewards.aerial import SimpleAerialReward, AirTimeReward, AirHeightReward
from rlrl.rewards.boost import VelocityPlayerToBoostReward
from rlrl.rewards.events import TouchReward, TeamGoalReward, ConcedeReward, DemoReward, BoostPickupReward
from rlrl.consts import GAME_TICK_RATE, TICK_SKIP

config = {
    "rewards": [
        ("touch", TouchReward(), 12),
        ("team_goal", TeamGoalReward(), 50),
        ("concede", ConcedeReward(), 50),
        ("boost_pickup", BoostPickupReward(), 8),
        ("demo", DemoReward(), 8),
        ("vel_player_to_ball", VelocityPlayerToBallReward(), 2.5),
        ("vel_ball_to_goal", VelocityBallToGoalReward(), 5),
        ("save_boost", SaveBoostReward(), 0.1),
        ("simple_aerial", SimpleAerialReward(), 4),
        ("air_time", AirTimeReward(), 0.3),
        ("air_height", AirHeightReward(), 0.4),
        ("vel_player_to_boost", VelocityPlayerToBoostReward(
            min_boost=0.7
        ), 2),
    ],
    "terminal_conditions": [
        GoalScoredCondition(),
        NoTouchTimeoutCondition(int(round(20 * GAME_TICK_RATE / TICK_SKIP))),
        TimeoutCondition(int(round(300 * GAME_TICK_RATE / TICK_SKIP))),
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
        "save_every_ts": 100_000,
    }
}
