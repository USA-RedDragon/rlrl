import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import common_values

class VelocityPlayerToBoostReward(RewardFunction):
    def __init__(self, min_boost=0.2):
        super().__init__()
        # If boost is normalized 0-1, 0.2 is 20 boost. 
        # If yours is 0-100, change to 20.
        self.min_boost = min_boost

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # If we have enough boost, don't encourage boost chasing
        if player.boost_amount >= self.min_boost:
            return 0.0

        # Logic to find closest active boost pad
        # state.boost_pads is usually an array of timers or booleans. 
        # Assuming 1 = active, 0 = inactive (standard RLGym)
        active_pad_indices = np.where(state.boost_pads == 1)[0]
        
        if len(active_pad_indices) == 0:
            return 0.0
            
        # Get positions of active pads
        active_pad_locs = np.array(common_values.BOOST_LOCATIONS)[active_pad_indices]
        
        # Find distance to all active pads
        # simple Euclidean distance
        dists = np.linalg.norm(active_pad_locs - player.car_data.position, axis=1)
        
        # Get index of closest pad
        closest_pad_idx = np.argmin(dists)
        target_pad_loc = active_pad_locs[closest_pad_idx]

        # Calculate velocity toward that pad
        vel = player.car_data.linear_velocity
        pos_diff = target_pad_loc - player.car_data.position
        
        norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
        norm_vel = vel / 2300 # Normalize by max speed
        
        # Return dot product
        return float(np.dot(norm_pos_diff, norm_vel))
