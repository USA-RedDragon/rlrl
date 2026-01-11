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
        ("save_boost", SaveBoostReward(), 5),
        ("aerial_touch_impulse", AerialTouchImpulseReward(
               min_impulse=200,
               impulse_scale=0.01
            ), 1),
        ("simple_aerial", SimpleAerialReward(), 2),
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
        "critic_layer_sizes": [256, 256, 256],
        "critic_lr": 2.0e-4,
        "exp_buffer_size": 120_000,
        "gae_gamma": 0.98,
        "gae_lambda": 0.95,
        "policy_layer_sizes": [256, 256, 256],
        "policy_lr": 1.5e-4,
        "ppo_batch_size": 50_000,
        "ppo_clip_range": 0.18,
        "ppo_ent_coef": 0.0018,
        "ppo_epochs": 3,
        "ppo_minibatch_size": 12_500,
        "save_every_ts": 1_000_000,
        "timestep_limit": 10e15,
        "ts_per_iteration": 25_000,
    }
}
