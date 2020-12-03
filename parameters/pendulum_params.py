PARAMS = {
    "action_range": 2,
    "alpha_actor": 0.0005,
    "alpha_critic": 0.001,
    "gamma": 0.99,
    "batch_size": 32,
    "buffer_size": 50000,
    "tau": 0.005,
    "train_begin": 2000,
    "episodes": 30000,
    "steps_per_episode": 300,
    "epochs": 13,
    "episodes_to_print": 100,
    "episodes_to_save": 300,
    "path": "./models/model",
    "load_path_critic": "./models/model_critic_model",
    "load_path_actor": "./models/model_actor_model",
    "reward_path": "./models/pendulum_reward.txt",
    "load_models": True
}
