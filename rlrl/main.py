import argparse
import os

import wandb

from rlgym_ppo import Learner

from rlrl.env import get_env_builder
from rlrl.metrics import Logger
from rlrl import sweep
from rlrl.consts import GAME_TICK_RATE, TICK_SKIP

def main():
    parser = argparse.ArgumentParser(description="Run PPO Agent")
    parser.add_argument("--n_proc", type=int, default=os.cpu_count(), help="Number of parallel environment processes.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training.")
    parser.add_argument("--config", type=str, default="rlrl.config.aerial", help="Python package path of the config to use, i.e. 'rlrl.config.aerial'.")
    parser.add_argument("--sweep_id", type=str, default=None, help="Sweep mode only: Existing W&B sweep ID to join.")
    parser.add_argument("--sweep_config", type=str, default=None, help="Sweep mode only: Python package path of the config to use, i.e. 'rlrl.config.sweeps.hyperparameters'.")
    parser.add_argument("--sweep_project", type=str, default="rlgym-ppo", help="Sweep mode only: W&B project name to use.")
    parser.add_argument("--render", action="store_true", help="Render one of the environments.")
    parser.add_argument('mode', nargs='?', default='train', help="Mode to run: 'train', 'sweep' or 'eval'.")
    args = parser.parse_args()

    config_module = __import__(args.config, fromlist=['config'])
    config = getattr(config_module, 'config')
    if config is None:
        raise ValueError(f"Config module {args.config} does not have a 'config' object.")

    learner_kwargs = {
        "n_proc": args.n_proc,
        "min_inference_size": args.n_proc,
        "random_seed": args.seed,
        "metrics_logger": Logger(reward_names=[name for name, _, _ in config.get("rewards", [])]),
        "render": args.render,
        "standardize_returns": True,
        "standardize_obs": False,
        "log_to_wandb": True,
        "load_wandb": True,
    }

    for key, value in config["hyperparameters"].items():
        learner_kwargs[key] = value

    if args.mode == 'train':
        Learner(get_env_builder(config), **learner_kwargs).learn()
    elif args.mode == 'eval':
        learner_kwargs["render_delay"] = TICK_SKIP / GAME_TICK_RATE
        learner_kwargs["log_to_wandb"] = False
        learner_kwargs["load_wandb"] = False
        learner_kwargs["n_proc"] = 1
        learner_kwargs["min_inference_size"] = 1
        learner_kwargs["ts_per_iteration"] = 1
        Learner(get_env_builder(config), **learner_kwargs).evaluate()
    elif args.mode == 'sweep':
        learner_kwargs["instance_launch_delay"] = 0.05

        sweep_id = args.sweep_id
        if sweep_id:
            print(f"Joining existing sweep {sweep_id}")
        else:
            if args.sweep_config is None:
                raise ValueError("Sweep config path must be provided when creating a new sweep.")
            sweep_config_module = __import__(args.sweep_config, fromlist=['config'])
            sweep_config_object = getattr(sweep_config_module, 'config')
            sweep_id = wandb.sweep(sweep_config_object, project=args.sweep_project)
            print(f"Created new sweep {sweep_id}")

        wandb.agent(sweep_id, function=sweep.get_sweep_trainer(learner_kwargs, config, args.sweep_project), count=None, project=args.sweep_project)
    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Must be 'train', 'eval' or 'sweep'.")

if __name__ == "__main__":
    main()
