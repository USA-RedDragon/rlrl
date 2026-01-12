import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import common_values

class InAirReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:        
        if not player.on_ground:
            return 1
        else:
            return 0

class SimpleAerialReward(RewardFunction):
    def __init__(self, min_height: float = 200.0):
        super().__init__()
        self.min_height = min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        # If player touches ball and is NOT on ground
        if player.ball_touched and not player.on_ground:
            # Reward based on height relative to ceiling.
            if player.car_data.position[2] > self.min_height:
                return player.car_data.position[2] / common_values.CEILING_Z
            
        return 0.0

class AerialPlayReward(RewardFunction):
    def __init__(self, ceiling_height: float = common_values.CEILING_Z, time_scale: float = 0.002, height_scale: float = 1.0):
        super().__init__()
        self.ceiling_height = ceiling_height
        self.time_scale = time_scale
        self.height_scale = height_scale
        self._airborne_steps = {}

    def reset(self, initial_state: GameState):
        self._airborne_steps.clear()

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        steps_airborne = self._airborne_steps.get(player.car_id, 0)

        if not player.on_ground:
            steps_airborne += 1
            self._airborne_steps[player.car_id] = steps_airborne
        else:
            self._airborne_steps[player.car_id] = 0
            return 0.0

        ball_height = max(0.0, state.ball.position[2])
        height_term = self.height_scale * min(ball_height / self.ceiling_height, 1.0)
        time_term = steps_airborne * self.time_scale

        return height_term * time_term


class AerialTouchImpulseReward(RewardFunction):
    def __init__(
        self,
        min_impulse: float = 100.0,
        impulse_scale: float = 0.001,
        min_boost_spent: float = 0.02,
        min_height: float = 200.0,
    ):
        super().__init__()
        self.min_impulse = min_impulse
        self.impulse_scale = impulse_scale
        self.min_boost_spent = min_boost_spent
        self.min_height = min_height
        self._prev_ball_vel = None
        self._prev_boost = None

    def reset(self, initial_state: GameState):
        self._prev_ball_vel = None
        self._prev_boost = None

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        ball_vel = np.asarray(state.ball.linear_velocity)

        reward = 0.0

        if player.car_data.position[2] < self.min_height:
            return 0.0

        if self._prev_ball_vel is not None and self._prev_boost is not None:
            boost_spent = self._prev_boost - player.boost_amount

            if (
                player.ball_touched
                and not player.on_ground
                and boost_spent > self.min_boost_spent
            ):
                delta_v = ball_vel - self._prev_ball_vel
                impulse = float(np.linalg.norm(delta_v))

                if impulse > self.min_impulse:
                    reward = (impulse - self.min_impulse) * self.impulse_scale

        self._prev_ball_vel = ball_vel
        self._prev_boost = player.boost_amount
        return reward
