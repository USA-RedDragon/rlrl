import numpy as np

from rlgym_sim.utils.reward_functions import RewardFunction
from rlgym_sim.utils.gamestates import GameState, PlayerData

class LoggingCombinedReward(RewardFunction):
    """Combined reward that also records per-component contributions on the GameState."""

    def __init__(self, reward_items):
        """
        reward_items: iterable of (name: str, reward_fn: RewardFunction, weight: float)
        """
        super().__init__()
        self.reward_items = []
        self.reward_names = []
        for name, reward_fn, weight in reward_items:
            self.reward_names.append(name)
            self.reward_items.append((name, reward_fn, weight))

    def reset(self, initial_state: GameState):
        for _, reward_fn, _ in self.reward_items:
            reward_fn.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        total = 0.0

        # Ensure per-step structures exist on the state for metrics logging
        breakdown = getattr(state, "_reward_breakdown", None)
        if not isinstance(breakdown, dict):
            breakdown = {}
            setattr(state, "_reward_breakdown", breakdown)

        player_rewards = breakdown.get(player.car_id)
        if player_rewards is None:
            player_rewards = {}
            breakdown[player.car_id] = player_rewards

        setattr(state, "_reward_names", self.reward_names)

        for name, reward_fn, weight in self.reward_items:
            value = reward_fn.get_reward(player, state, previous_action)
            contrib = weight * value
            total += contrib
            player_rewards[name] = contrib

        return total
