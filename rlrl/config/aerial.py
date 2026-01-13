from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward, SaveBoostReward
from rlgym_sim.utils.state_setters.random_state import RandomState
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition

from rlrl.rewards.aerial import AerialTouchImpulseReward, SimpleAerialReward
from rlrl.rewards.boost import VelocityPlayerToBoostReward
from rlrl.consts import GAME_TICK_RATE, TICK_SKIP

config = {
    "rewards": [
        ("event", EventReward(
               touch=10,
               team_goal=25,
               concede=-25,
               boost_pickup=8,
               demo=8,
            ), 1),
        ("vel_player_to_ball", VelocityPlayerToBallReward(), 5),
        ("vel_ball_to_goal", VelocityBallToGoalReward(), 8),
        ("save_boost", SaveBoostReward(), 2),
        ("aerial_touch_impulse", AerialTouchImpulseReward(
               min_impulse=200,
               impulse_scale=0.005
            ), 1),
        ("simple_aerial", SimpleAerialReward(), 8),
        ("vel_player_to_boost", VelocityPlayerToBoostReward(
               min_boost=0.5
            ), 4),
    ],
    "terminal_conditions": [
        GoalScoredCondition(),
        NoTouchTimeoutCondition(int(round(20 * GAME_TICK_RATE / TICK_SKIP)))
    ],
    "state_setter": RandomState(ball_rand_speed=False, cars_rand_speed=True, cars_on_ground=False),
    "hyperparameters": {
        "timestep_limit": 10e15,
        "exp_buffer_size": 12_500_000,
        "ts_per_iteration": 5000,
        "sac_policy_delay": 3,
        "sac_batch_size": 1024,
        "sac_ent_coef": 'auto',
        "sac_learning_rate": 3e-4,
        "sac_learning_starts": 10_000,
        "sac_train_freq": 5,
        "sac_gradient_steps": 1,
        "sac_tau": 0.005,
        "sac_gamma": 0.99,
        "policy_layer_sizes": (256, 256),
        "critic_layer_sizes": (256, 256),
        "save_every_ts": 1_000_000,
    }
}
