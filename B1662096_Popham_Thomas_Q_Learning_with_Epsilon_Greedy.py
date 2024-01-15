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


# Initialise the environments with the necessary environment parameters set

# Frozen Lake Environment - 4x4
frozen_maze_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)

# Cliff Walking Environment
cliff_walking_env = gym.make('CliffWalking-v0')

# Taxi Environment
taxi_env = gym.make('Taxi-v3')



# Start training based on the environment's action sample and the necessary hyperparameters
def trainWithRandomActions(env, num_obs, num_steps, learning_rate, gamma, max_epsilon, min_epsilon, prob_decay):

  # Check the actions, reward range and observation space for the environment that is passed
  print("Number of possible actions to take in environment: {}".format(env.action_space.n))
  print("Number of possible states in environment: {}".format(env.observation_space.n))
  print("Action space: {}".format(env.action_space))
  print("Observation space: {}".format(env.observation_space))
  print("Observation space shape: {}".format(env.observation_space.shape))

  # Create a Q table to track the number of states and actions that are required by the model based on the environment
  q_table = np.zeros((env.observation_space.n, env.action_space.n))

  total_obs_rewards = []

  epsilon = max_epsilon

  for obs in range(num_obs):
    # Reset the environment before stepping through it for every observation
    init_obs = env.reset()
    # Define the rewards list that gains the rewards from the steps made
    obs_rewards = []
    
    # Step through the environment and retrieve observations from the environment
    for step in range(num_steps):

        # Visualise the environment as a pop-up window
        env.render()

        # Randomly generate a number from 0 to 1
        random_int_value = random.uniform(0,1)

        # If the random number is greater than the epsilon value
        if random_int_value < epsilon:
          # Create a random action to pass when making each step
          action = env.action_space.sample()
        else:
          # Get the action on the highest value based on the state
          action = np.argmax(q_table[init_obs, :])
      
        # Check the action sample gained from the epsilon greedy method
        print("Action Sample: {}".format(action))

        # For every step through the environment, retrieve the observation, reward, if its done and environment info
        new_obs, reward, done, info = env.step(action)
        # Check the contents of the steps taken
        print("Observations: {}".format(new_obs))
        print("Observations Data Type: {}".format(type(new_obs)))
        print("Reward: {}".format(reward))
        print("If done: {}".format(done))
        print("Environment info: {}".format(info))

        # Check the probability of the state transitioning to another state
        print("Transition Probability based on the environment {}".format(env.P[new_obs][action]))

        # Update the Q table with the action and observation (state) values
        q_table[init_obs][action] = q_table[init_obs][action] + learning_rate * (reward + gamma * np.max(q_table[new_obs, :]) - q_table[init_obs][action])

        # Add the reward based on the environment and its state
        obs_rewards.append(reward)

        # Make sure that the state is always updated when stepping through the envinroment
        obs = new_obs

        # Check the contents of the q table based on the random actions based on the environment's action space
        with np.printoptions(precision=5, suppress=True):
          print("Q Table After Random Sampling: {}".format(q_table))

        if done:
          total_obs_rewards.append(obs_rewards)
          # Reduce the value of epsilon, to ensure that the agent explores a good amount of state space
          epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-prob_decay*obs)
          break

  # Close the environment
  env.close()
  
  # Return the q_table
  return q_table


# Start training based on the environment's action sample and the necessary hyperparameters
def evaluateTrainedQTable(env, num_obs, num_steps, q_table):

  # Check the actions, reward range and observation space for the environment that is passed
  print("Number of possible actions to take in environment: {}".format(env.action_space.n))
  print("Number of possible states in environment: {}".format(env.observation_space.n))
  print("Action space: {}".format(env.action_space))
  print("Observation space: {}".format(env.observation_space))
  print("Observation space shape: {}".format(env.observation_space.shape))

  total_obs_rewards = []

  for obs in range(num_obs):
    # Reset the environment before stepping through it for every observation
    init_obs = env.reset()
    # Define the rewards list that gains the rewards from the steps made
    obs_rewards = 0
    
    # Step through the environment and retrieve observations from the environment
    # Whether the agent fails to reach the goal in each observation and returns done
    for step in range(num_steps):
        # Visualise the environment as a pop-up window
        env.render()
        # Get an action from the greedy method
        action = np.argmax(q_table[init_obs, :])
        # Check the action sample gained from the epsilon greedy method
        print("Action Sample: {}".format(action))
        # For every step through the environment, retrieve the observation, reward, if its done and environment info
        new_obs, reward, done, info = env.step(action)
        # Check the contents of the steps taken
        print("Observations: {}".format(new_obs))
        print("Observations Data Type: {}".format(type(new_obs)))
        print("Reward: {}".format(reward))
        print("If done: {}".format(done))
        print("Environment info: {}".format(info))

        # Check the probability of the state transitioning to another state
        print("Transition Probability based on the environment {}".format(env.P[init_obs][action]))

        # Add the reward based on the environment and its state
        obs_rewards += reward

        if done:
          # Accumulate all the rewards from the observation to the overall total
          total_obs_rewards.append(obs_rewards)
          if reward == 1:
            print("Goal reached")
          else:
            print("Failed to reach the goal")

  # Close the environment
  env.close()

  # Overall success rate
  print('Overall Success rate: {0:.2f} %'.format(100*np.sum(total_obs_rewards)/len(total_obs_rewards)))

  # Average number of steps to take when reaching the goal
  print('Average number of steps taken to reach the goal: {0:.2f}'.format(np.mean(num_steps)))

  # Get the overall mean and std of all the gained rewards from each observation
  mean_reward = np.mean(total_obs_rewards)
  std_reward = np.std(total_obs_rewards)
  
  # Return the mean_reward and the std reward
  return mean_reward, std_reward


# Set the inital learning rate and gamma for the model and custom evaluation method
learning_rate = 0.8
num_obs = 1000
num_steps = 124
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.01
prob_decay = 0.001

# Train the environment with specific number of observations
trained_Q_table = trainWithRandomActions(frozen_maze_env, num_obs, num_steps, learning_rate, gamma, max_epsilon, min_epsilon, prob_decay)
# Print mean reward from the custom training method
print("Trained Q Table from random action sample - Frozen Lake:- {}".format(trained_Q_table))



# Evaluate the environment with specific number of observations
Q_frozen_lake_mean_reward, std_reward = evaluateTrainedQTable(frozen_maze_env, 1000, num_steps, trained_Q_table)
# Print mean reward from the custom training method
print("Mean Reward from random action sample - Frozen Lake:- {}".format(Q_frozen_lake_mean_reward))
print("\n")
# Print std reward from the custom training method
print("Std Reward from random action sample - Frozen Lake:- {}".format(std_reward))



# Set the inital learning rate and gamma for the model and custom training method
learning_rate = 0.8
num_obs = 1000
num_steps = 100
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.01
prob_decay = 0.001

# Train the environment with specific number of observations
trained_Q_table = trainWithRandomActions(cliff_walking_env, num_obs, num_steps, learning_rate, gamma, max_epsilon, min_epsilon, prob_decay)
print("Q Table After Training {}".format(trained_Q_table))



Q_cliff_walking_mean_reward, std_reward = evaluateTrainedQTable(cliff_walking_env, 1000, num_steps, learning_rate, trained_Q_table)

# Print mean reward from the custom evaluation method
print("Mean Reward from random action sample - Cliff Walking:- {}".format(Q_cliff_walking_mean_reward))
print("\n")
# Print std reward from the custom evaluation method
print("Std Reward from random action sample - Cliff Walking:- {}".format(std_reward))



# Set the inital learning rate and gamma for the model and custom training method
learning_rate = 0.8
num_obs = 1000
num_steps = 100
gamma = 0.95
max_epsilon = 1.0
min_epsilon = 0.01
prob_decay = 0.001

# Train the environment with specific number of observations
trained_Q_table = trainWithRandomActions(taxi_env, num_obs, num_steps, learning_rate, gamma, max_epsilon, min_epsilon, prob_decay)
print("Q Table After Training {}".format(trained_Q_table))


Q_taxi_mean_reward, std_reward = evaluateTrainedQTable(taxi_env, num_obs, 1000, learning_rate, trained_Q_table)

# Print mean reward from the custom evaluation method
print("Mean Reward from random action sample - Taxi:- {}".format(Q_taxi_mean_reward))
print("\n")
# Print std reward from the custom evaluation method
print("Std Reward from random action sample - Taxi:- {}".format(std_reward))


# Visualise the overall results from each of the Q Learning rewards gained on each of the 3 environments
sns.set_theme()
Q_Learning_Environment_Labels = ['Frozen Lake', 'Cliff Walking', 'Taxi']
x_labels = np.arange(len(Q_Learning_Environment_Labels))
width = 0.35
fig, ax = plt.subplots()
Q_learning_rect1 = ax.bar(x_labels - width/3, Q_frozen_lake_mean_reward, width, label='Q Learning Mean Score - Frozen Lake')
Q_learning_rect2 = ax.bar(x_labels - width/3, Q_cliff_walking_mean_reward, width, label='Q Learning Mean Score - Cliff Walking')
Q_learning_rect3 = ax.bar(x_labels - width/3, Q_taxi_mean_reward, width, label='Q Learning Mean Score - Taxi')

ax.set_ylabel('Mean Scores')
ax.set_title('Q Learning - Overall Mean Scores')
ax.set_xticks(x_labels, Q_Learning_Environment_Labels)
ax.legend()

ax.bar_label(Q_learning_rect1, padding=3)
ax.bar_label(Q_learning_rect2, padding=3)
ax.bar_label(Q_learning_rect3, padding=3)

fig.tight_layout()

plt.show()







