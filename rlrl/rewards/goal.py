from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData
from rlgym_sim.utils import common_values

class OwnGoalReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self._score = 0

    def reset(self, initial_state: GameState):
        self._score = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.team_num == common_values.BLUE_TEAM and state.orange_score > self._score:
            self._score = state.orange_score
            return -1.0
        elif player.team_num == common_values.ORANGE_TEAM and state.blue_score > self._score:
            self._score = state.blue_score
            return -1.0
        return 0.0
