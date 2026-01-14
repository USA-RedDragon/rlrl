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


class AirTimeReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.on_ground:
            return 0.0

        return 1.0


class AirHeightReward(RewardFunction):
    def __init__(self, ceiling_height: float = common_values.CEILING_Z):
        super().__init__()
        self.ceiling_height = ceiling_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.on_ground:
            return 0.0

        # Reward height, but cap it at the ball's height. 
        # Encourages climbing to the ball, but not flying uselessly high.
        player_height = player.car_data.position[2]
        ball_height = state.ball.position[2]
        
        return min(player_height, ball_height) / self.ceiling_height



class AerialTouchImpulseReward(RewardFunction):
    def __init__(
        self,
        min_impulse: float = 100.0,
        min_boost_spent: float = 0.02,
        min_height: float = 200.0,
    ):
        super().__init__()
        self.min_impulse = min_impulse
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
                    # Normalize impulse by max car speed + max ball speed approx
                    reward = impulse / (2300 + 6000)

        self._prev_ball_vel = ball_vel
        self._prev_boost = player.boost_amount
        return min(reward, 1.0)
