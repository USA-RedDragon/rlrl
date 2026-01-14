import numpy as np
from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils.common_values import BLUE_TEAM

class TouchReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_touched = {}

    def reset(self, initial_state: GameState):
        self.last_touched = {}
        for p in initial_state.players:
            self.last_touched[p.car_id] = p.ball_touched

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        old_touched = self.last_touched.get(player.car_id, False)
        new_touched = player.ball_touched
        
        reward = 0.0
        # Replicating EventReward logic: only reward when value increases (False -> True)
        if new_touched and not old_touched:
            reward = 1.0
            
        self.last_touched[player.car_id] = new_touched
        return reward

class TeamGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_score = {}

    def reset(self, initial_state: GameState):
        self.last_score = {}
        for p in initial_state.players:
            self.last_score[p.car_id] = initial_state.blue_score if p.team_num == BLUE_TEAM else initial_state.orange_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        curr_score = state.blue_score if player.team_num == BLUE_TEAM else state.orange_score
        old_score = self.last_score.get(player.car_id, curr_score)
        
        reward = 0.0
        if curr_score > old_score:
            reward = float(curr_score - old_score)
            
        self.last_score[player.car_id] = curr_score
        return reward

class ConcedeReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_score = {}

    def reset(self, initial_state: GameState):
        self.last_score = {}
        for p in initial_state.players:
            self.last_score[p.car_id] = initial_state.orange_score if p.team_num == BLUE_TEAM else initial_state.blue_score

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        curr_score = state.orange_score if player.team_num == BLUE_TEAM else state.blue_score
        old_score = self.last_score.get(player.car_id, curr_score)
        
        reward = 0.0
        if curr_score > old_score:
            reward = float(curr_score - old_score)
            
        self.last_score[player.car_id] = curr_score
        return reward

class DemoReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_demos = {}

    def reset(self, initial_state: GameState):
        self.last_demos = {}
        for p in initial_state.players:
            self.last_demos[p.car_id] = p.match_demolishes

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        curr_demos = player.match_demolishes
        old_demos = self.last_demos.get(player.car_id, curr_demos)
        
        reward = 0.0
        if curr_demos > old_demos:
            reward = float(curr_demos - old_demos)
            
        self.last_demos[player.car_id] = curr_demos
        return reward

class BoostPickupReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_boost = {}

    def reset(self, initial_state: GameState):
        self.last_boost = {}
        for p in initial_state.players:
            self.last_boost[p.car_id] = p.boost_amount

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        curr_boost = player.boost_amount
        old_boost = self.last_boost.get(player.car_id, curr_boost)
        
        reward = 0.0
        if curr_boost > old_boost:
            reward = curr_boost - old_boost
            
        self.last_boost[player.car_id] = curr_boost
        return reward
