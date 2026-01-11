"""
W&B sweep runner for the current project layout.
Uses config.py/env.py/metrics.py just like main.py, but with sweep-driven overrides.
"""

import wandb
from rlgym_ppo import Learner

from rlrl.env import get_env_builder

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

def get_sweep_trainer(learner_kwargs, config, wandb_project):
    def sweep_trainer(wandb_cfg=None):
        run = wandb.init(project=wandb_project, config=wandb_cfg, name=None, reinit=True)
        run.name = run.id
        cfg = wandb.config

        for key, value in cfg.items():
            # only override existing keys
            if key in learner_kwargs:
                learner_kwargs[key] = value

        learner = Learner(
            get_env_builder(config),
            wandb_run=run,
            wandb_run_name=run.name,
            checkpoints_save_folder=f"data/checkpoints/{run.id}",
            **learner_kwargs,
        )

        print(f"\n=== Starting W&B run: {run.name} ===\n")
        learner.learn()
        print(f"\n=== Completed W&B run: {run.name} ===\n")

    return sweep_trainer
