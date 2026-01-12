config = {
    "method": "random",
    "metric": {"name": "Policy Reward", "goal": "maximize"},
    "parameters": {
        "random_seed": {"values": [1, 2, 3]},
        "exp_buffer_size": {"min": 100_000, "max": 50_000_000 },
        "ts_per_iteration": {"min": 5, "max": 50_000 },
        "sac_batch_size": {"min": 128, "max": 5_000_000 },
        "sac_max_updates_per_iter": {"min": 500, "max": 5_000_000 },
        "sac_ent_coef": {"min": 0.01, "max": 0.2},
        "sac_learning_rate": {"min": 1e-4, "max": 1e-3},
        "sac_learning_starts": {"min": 5_000, "max": 200_000},
        "sac_train_freq": {"min": 4, "max": 8},
        "sac_tau": {"min": 0.001, "max": 0.01},
        "sac_gamma": {"min": 0.95, "max": 0.999},
        "policy_layer_sizes": {"values": [(256, 256), (256, 256, 256), (512, 512)]},
        "critic_layer_sizes": {"values": [(256, 256), (256, 256, 256), (512, 512)]},
    }
}
