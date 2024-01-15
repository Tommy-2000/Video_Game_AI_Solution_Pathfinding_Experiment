# Core Python Libraries
import os
import sys
import math
import random
import gzip
from queue import PriorityQueue
from timeit import timeit

# Anaconda/Pip Python Libraries
from IPython import display
import numpy as np
import pandas as pd
import scipy as sci
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Reinforcement Learning Python Libraries
import gym as gym
import pygame
import stable_baselines3 as sb3
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# Set a reward-based callback method for every instance of training an algorithm
# When the evaluation reaches its maximum reward, the algorithm (for example A2C) should stop training
reward_stop_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)



# Create callback helper methods from Stable Baselines 3 to properly evaluate each environment with each model under specific conditions
frozen_maze_eval_callback = EvalCallback(frozen_maze_env, deterministic=False, best_model_save_path=saved_models_path, log_path=training_log_path, n_eval_episodes=10000)

cliff_walking_eval_callback = EvalCallback(cliff_walking_env, deterministic=False, best_model_save_path=saved_models_path, log_path=training_log_path, n_eval_episodes=10000)

taxi_eval_callback = EvalCallback(taxi_env, deterministic=False, best_model_save_path=saved_models_path, log_path=training_log_path, n_eval_episodes=10000)



# Vectorize the environment in order to work with Stable Baselines during training
vec_frozen_maze_env = DummyVecEnv([lambda:frozen_maze_env])

# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 2.5e-4
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
prob_decay = 0.0005

# Perform the same training and evaluation with the PPO model
ppo_model_frozen_maze = PPO("MlpPolicy", vec_frozen_maze_env, verbose=1, learning_rate=learning_rate, gamma=gamma, batch_size=128, clip_range=0.3, n_steps=128, max_grad_norm=0.9, vf_coef=0.045, ent_coef=1e-7)



# Apply the untrained model with the environment over a certain number of observations
mean_reward, std_reward = evaluate_policy(model=ppo_model_frozen_maze, env=vec_frozen_maze_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", sum(mean_reward), "Number of observations made: ", num_obs)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", num_obs)


# Train the model with the specified number of timesteps
trained_ppo_model_frozen_maze = ppo_model_frozen_maze.learn(total_timesteps=30000, callback=frozen_maze_eval_callback)


# Apply the trained model with the environment and the evaluation callback method
ppo_mean_reward_frozen_maze, std_reward = evaluate_policy(model=trained_ppo_model_frozen_maze, env=vec_frozen_maze_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", sum(ppo_mean_reward_frozen_maze), "Number of observations made: ", 10000)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", 10000)


# Vectorize the environment in order to work with Stable Baselines during training
vec_cliff_walking_env = DummyVecEnv([lambda:cliff_walking_env])

# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 2.5e-4
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
prob_decay = 0.0005

# Perform the same training and evaluation with the PPO model
ppo_model_cliff_walking = PPO("MlpPolicy", vec_cliff_walking_env, verbose=1, learning_rate=learning_rate, gamma=gamma, batch_size=128, clip_range=0.3, n_steps=128, max_grad_norm=0.9, vf_coef=0.045, ent_coef=0.04)


mean_reward, std_reward = evaluate_policy(model=ppo_model_cliff_walking, env=vec_cliff_walking_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", sum(mean_reward), "Number of observations made: ", 10000)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", 10000)


# Train the model with the specified number of timesteps
trained_ppo_model_cliff_walking = ppo_model_cliff_walking.learn(total_timesteps=30000, callback=cliff_walking_eval_callback)


# Apply the trained model with the environment and the evaluation callback method
ppo_mean_reward_cliff_walking, std_reward = evaluate_policy(model=trained_ppo_model_cliff_walking, env=vec_cliff_walking_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", sum(ppo_mean_reward_cliff_walking), "Number of observations made: ", num_obs)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", num_obs)


# Vectorize the environment in order to work with Stable Baselines during training
vec_taxi_env = DummyVecEnv([lambda:taxi_env])

# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 2.5e-4
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
prob_decay = 0.0005

# Perform the same training and evaluation with the PPO model
ppo_model_taxi = PPO("MlpPolicy", vec_taxi_env, verbose=1, learning_rate=learning_rate, gamma=gamma, batch_size=128, clip_range=0.3, n_steps=128, max_grad_norm=0.9, create_eval_env=True, vf_coef=0.045, ent_coef=0.04))

# Apply the untrained model with the environment over a certain number of observations
mean_reward, std_reward = evaluate_policy(model=ppo_model_taxi, env=vec_taxi_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", mean_reward, "Number of observations made: ", 10000)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", 10000)


# Train the model with the specified number of timesteps
trained_ppo_model_taxi = ppo_model_taxi.learn(total_timesteps=30000, callback=taxi_eval_callback)



# Apply the trained model with the environment and the evaluation callback method
ppo_mean_reward_taxi, std_reward = evaluate_policy(model=trained_ppo_model_taxi, env=vec_taxi_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", sum(ppo_mean_reward_taxi), "Number of observations made: ", 10000)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", 10000)



# Vectorize the environment in order to work with Stable Baselines during training
vec_frozen_maze_env = DummyVecEnv([lambda:frozen_maze_env])

# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 2.5e-4
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
prob_decay = 0.0005

# Perform the same training and evaluation with the DQN model
dqn_model_frozen_maze = DQN("MlpPolicy", vec_frozen_maze_env, verbose=1, buffer_size=50000, learning_rate=learning_rate, batch_size=32, max_grad_norm=0.9)


# Apply the untrained model with the environment over a certain number of observations
mean_reward, std_reward = evaluate_policy(model=dqn_model_frozen_maze, env=vec_frozen_maze_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", mean_reward, "Number of observations made: ", 10000)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", 10000)


# Train the model with the specified number of timesteps
trained_dqn_model_frozen_maze = dqn_model_frozen_maze.learn(total_timesteps=30000, callback=frozen_maze_eval_callback)



# Apply the trained model with the environment over a certain number of observations
dqn_mean_reward_frozen_maze, std_reward = evaluate_policy(model=trained_dqn_model_frozen_maze, env=vec_frozen_maze_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", dqn_mean_reward_frozen_maze, "Number of observations made: ", 10000)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", 10000)


# Vectorize the environment in order to work with Stable Baselines during training
vec_cliff_walking_env = DummyVecEnv([lambda:cliff_walking_env])

# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 2.5e-4
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
prob_decay = 0.0005

# Perform the same training and evaluation with the DQN model
dqn_model_cliff_walking = DQN("MlpPolicy", vec_cliff_walking_env, verbose=1, buffer_size=50000, learning_rate=learning_rate, batch_size=32, max_grad_norm=0.9)reate_eval_env=True, vf_coef=0.045, ent_coef=0.04)



# Apply the untrained model with the environment over a certain number of observations
mean_reward, std_reward = evaluate_policy(model=dqn_model_cliff_walking, env=vec_cliff_walking_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", mean_reward, "Number of observations made: ", num_obs)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", num_obs)



# Train the model with the specified number of timesteps
trained_dqn_model_cliff_walking = dqn_model_cliff_walking.learn(total_timesteps=30000, callback=cliff_walking_eval_callback)



# Apply the trained model with the environment over a certain number of observations
dqn_mean_reward_cliff_walking, std_reward = evaluate_policy(model=trained_dqn_model_cliff_walking, env=vec_cliff_walking_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", dqn_mean_reward_cliff_walking, "Number of observations made: ", num_obs)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", num_obs)



# Vectorize the environment in order to work with Stable Baselines during training
vec_taxi_env = DummyVecEnv([lambda:taxi_env])

# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 5e-4
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.05
prob_decay = 0.0005

# Perform the same training and evaluation with the DQN model
dqn_model_taxi = DQN("MlpPolicy", vec_taxi_env, verbose=1, buffer_size=50000, learning_rate=learning_rate, batch_size=32, max_grad_norm=0.9)


# Apply the untrained model with the environment over a certain number of observations
mean_reward, std_reward = evaluate_policy(model=dqn_model_taxi, env=vec_taxi_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", mean_reward, "Number of observations made: ", num_obs)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", num_obs)


# Train the model with the specified number of timesteps
trained_dqn_model_taxi = dqn_model_taxi.learn(total_timesteps=30000, callback=taxi_eval_callback)


# Apply the trained model with the environment over a certain number of observations
dqn_mean_reward_taxi, std_reward = evaluate_policy(model=trained_dqn_model_taxi, env=vec_taxi_env, deterministic=False, warn=True, render=True, return_episode_rewards=True, n_eval_episodes=10000)

print("Mean Reward: ", dqn_mean_reward_taxi, "Number of observations made: ", num_obs)
print("\n")
print("Std Reward: ", std_reward, "Number of observations made: ", num_obs)


# Visualise the overall results from each of the Stable Baselines 3 models on each of the 3 environments
sns.set_theme()
SB3_Models = ['A2C', 'PPO', 'DQN']
ppo_mean_scores = [sum(ppo_mean_reward_frozen_maze), sum(ppo_mean_reward_cliff_walking), sum(ppo_mean_reward_taxi)]
dqn_mean_scores = [sum(dqn_mean_reward_frozen_maze), sum(dqn_mean_reward_cliff_walking), sum(dqn_mean_reward_taxi)]

x_labels = np.arange(len(SB3_Models))
width = 0.35
fig, ax = plt.subplots()
ppo_rect = ax.bar(x_labels - width/3, ppo_mean_scores, width, label='PPO Mean Scores')
dqn_rect = ax.bar(x_labels - width/3, dqn_mean_scores, width, label='DQN Mean Scores')

ax.set_ylabel('Mean Scores')
ax.set_title('Stable Baselines 3 Scores - After Training')
ax.set_xticks(x_labels, SB3_Models)
ax.legend()

ax.bar_label(ppo_rect, padding=3)
ax.bar_label(dqn_rect, padding=3)

fig.tight_layout()

plt.show()
