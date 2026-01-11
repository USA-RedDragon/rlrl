import numpy as np

from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState

class Logger(MetricsLogger):
    def __init__(self, reward_names=None):
        self.current_run_metrics = {}
        self._prev_ball_vel = None
        self._step_count = 0
        self._max_aerial_impulse = 0.0
        self._boost_sum = 0.0
        self._reward_names = list(reward_names) if reward_names else []
        self._reward_sums = {name: 0.0 for name in self._reward_names}

    def _collect_metrics(self, game_state: GameState, done: bool) -> list:
        reward_names = getattr(game_state, "_reward_names", self._reward_names)
        if reward_names and self._reward_names != list(reward_names):
            self._reward_names = list(reward_names)
            self._reward_sums = {name: 0.0 for name in self._reward_names}

        if len(game_state.players) == 0:
            return [0.0] * (8 + len(self._reward_names))
        
        total_dist = 0
        total_boost = 0.0
        reward_breakdown = getattr(game_state, "_reward_breakdown", {})

        if self._prev_ball_vel is None:
            self._prev_ball_vel = np.asarray(game_state.ball.linear_velocity)

        self._step_count += 1

        for p in game_state.players:
            if p.car_id not in self.current_run_metrics:
                self.current_run_metrics[p.car_id] = {"touched": False, "aerial": False}

            # Check touch
            if p.ball_touched:
                self.current_run_metrics[p.car_id]["touched"] = True
                # Aerial check: car height > 200 (approx 2 ball widths) to properly filter out ground dodges
                if p.car_data.position[2] > 200:
                    self.current_run_metrics[p.car_id]["aerial"] = True

                # Track aerial touch impulse via ball velocity change
                ball_vel = np.asarray(game_state.ball.linear_velocity)
                impulse = float(np.linalg.norm(ball_vel - self._prev_ball_vel))
                if impulse > self._max_aerial_impulse:
                    self._max_aerial_impulse = impulse

            # Distance
            p_pos = p.car_data.position
            ball_pos = game_state.ball.position
            dist = np.linalg.norm(p_pos - ball_pos)
            total_dist += dist

            # Track average boost level across players per step
            total_boost += getattr(p, "boost_amount", 0.0)

        avg_dist = total_dist / len(game_state.players)
        avg_boost_step = total_boost / len(game_state.players)
        self._boost_sum += avg_boost_step

        avg_rewards_step = []
        for name in self._reward_names:
            step_total = 0.0
            for rewards in reward_breakdown.values():
                step_total += rewards.get(name, 0.0)
            avg_step = step_total / len(game_state.players)
            avg_rewards_step.append(avg_step)
            self._reward_sums[name] = self._reward_sums.get(name, 0.0) + avg_step

        # Update prev ball velocity for next step
        self._prev_ball_vel = np.asarray(game_state.ball.linear_velocity)

        if done:
            n_players = len(game_state.players)
            total_touched = 0
            total_aerial = 0
            total_goals = 0

            for p in game_state.players:
                metrics = self.current_run_metrics.get(p.car_id, {"touched": False, "aerial": False})
                if metrics["touched"]:
                    total_touched += 1
                if metrics["aerial"]:
                    total_aerial += 1
                total_goals += p.match_goals

            stats = [1.0,
                     total_touched / n_players,
                     total_aerial / n_players,
                     float(total_goals) / n_players,
                     avg_dist,
                     float(self._step_count),
                     self._max_aerial_impulse,
                     self._boost_sum / self._step_count if self._step_count > 0 else 0.0]
            if self._reward_names:
                reward_avgs = [self._reward_sums[name] / self._step_count if self._step_count > 0 else 0.0 for name in self._reward_names]
                stats.extend(reward_avgs)
            # Reset
            self.current_run_metrics = {}
            self._prev_ball_vel = None
            self._step_count = 0
            self._max_aerial_impulse = 0.0
            self._boost_sum = 0.0
            self._reward_sums = {name: 0.0 for name in self._reward_names}
            return stats
        else:
            base = [0.0, 0.0, 0.0, 0.0, avg_dist, 0.0, 0.0, avg_boost_step]
            if self._reward_names:
                base.extend(avg_rewards_step)
            return base

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        if not collected_metrics:
            return

        # collected_metrics is a list of lists of 1-element arrays; flatten each metric row
        rows = []
        expected_len = None
        for metrics_row in collected_metrics:
            flat = [float(x) for x in metrics_row]
            if len(flat) >= 8:
                if expected_len is None:
                    expected_len = len(flat)
                if len(flat) == expected_len:
                    rows.append(flat)

        if not rows:
            return

        data = np.vstack(rows)
        
        # Column 0: is_run_end
        # Column 1: run_touched (only valid if is_run_end)
        # Column 2: run_aerial (only valid if is_run_end)
        # Column 3: goals (only valid if is_run_end, or accumulated?)
        # Column 4: distance
        # Column 7: boost amount (per-step average or run average on final row)
        # Columns 8+: per-reward averages
        
        n_runs = np.sum(data[:, 0])
        n_touched = np.sum(data[:, 1])
        n_aerial = np.sum(data[:, 2])
        total_goals = np.sum(data[:, 3])

        avg_dist = np.mean(data[:, 4])
        avg_boost = np.mean(data[:, 7])

        base_cols = 8
        n_reward_cols = data.shape[1] - base_cols if data.shape[1] > base_cols else 0
        reward_means = []
        if n_reward_cols > 0:
            reward_means = np.mean(data[:, base_cols:], axis=0)

        # Use only completed runs for episode length and impulse stats
        completed_mask = data[:, 0] == 1
        if completed_mask.any():
            avg_episode_len = float(np.mean(data[completed_mask, 5]))
            avg_aerial_impulse = float(np.mean(data[completed_mask, 6]))
        else:
            avg_episode_len = 0.0
            avg_aerial_impulse = 0.0

        hit_rate = n_touched / n_runs if n_runs > 0 else 0
        aerial_rate = n_aerial / n_runs if n_runs > 0 else 0
        goals_per_run = total_goals / n_runs if n_runs > 0 else 0

        report = {
            "Hit Percentage": hit_rate,
            "Aerial Hit Percentage": aerial_rate,
            "Goals per Run": goals_per_run,
            "Avg Distance to Ball": avg_dist,
            "Episode Length (steps)": avg_episode_len,
            "Aerial Touch Impulse": avg_aerial_impulse,
            "Average Boost": avg_boost,
        }

        for idx, value in enumerate(reward_means):
            name = self._reward_names[idx] if idx < len(self._reward_names) else f"reward_{idx}"
            report[f"Reward/{name}"] = float(value)
        wandb_run.log(report)
