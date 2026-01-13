config = {
    "method": "random",
    "metric": {"name": "Policy Reward", "goal": "maximize"},
    "parameters": {
        "random_seed": {"values": [1, 2, 3]},
        "exp_buffer_size": {"min": 1_000_000, "max": 10_000_000 },
        "sac_max_updates_per_iter": {"min": 2_500, "max": 30_000 },
        "sac_ent_coef": {"min": 0.01, "max": 0.2},
    }
}
