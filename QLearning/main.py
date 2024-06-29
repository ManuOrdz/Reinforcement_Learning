#Imports and Environment Setup
from __future__ import annotations

from collections import defaultdict

import gymnasium as gym

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from Agent import QLearning
# Create the enviroment

env = gym.make("MountainCar-v0", render_mode='rgb_array')

#Observing the enviroment

observation, info = env.reset()
print(f'Observation:{observation}')
print(f'Information:{info}')

#sample a random action from all valid actions
action_space = env.action_space
observation_space = env.observation_space
print(f'Action space: {action_space}')
print(f'Observation space: {observation_space}')
action = env.action_space.sample()
#execute the action in our enviroment and receive infos from the environment

observation, reward, terminated, truncated, info = env.step(action)
print("New state obtained from action:",action)
print(f'Observation:{observation} \nReward:{reward}\nTerminated:{terminated}')

##hiperparameters
learning_rate = 0.05
n_episodes = 100000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes/2)
final_epsilon = 0.1

env = gym.wrappers.RecordVideo (env=env, 
                                video_folder='QLearning/tmp',
                                name_prefix='test-video',
                                episode_trigger=lambda x: (x+1)%30000 == 0)

env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=n_episodes)



obs_min = env.observation_space.low
obs_max = env.observation_space.high

agent = QLearning(
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    state, info = env.reset()
    state = agent.map_obs_to_stat(state, obs_max, obs_min)
    done = False
    while not done:
        action = agent.get_action(env.action_space, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = agent.map_obs_to_stat(next_state, obs_max, obs_min)
        #update the agent
        agent.update(state, action, reward, next_state)
        #update if the environment is done and the current state
        
        done = terminated or truncated
        state = next_state
            
    agent.decay_epsilon()




rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12,5))
axs[0].set_title("Episode rewards")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(), np.ones(rolling_length), mode='valid'
    )
    / rolling_length
)

axs[0].plot(range(len(reward_moving_average)),reward_moving_average)
axs[1].set_title("Episode lengths")

length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(), np.ones(rolling_length), mode='same'
    )
    / rolling_length
)

axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[2].set_title("Training Error")

training_error_moving_average = (
    np.convolve(
        np.array(agent.training_error), np.ones(rolling_length), mode='same'
    )
    /rolling_length
)

axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
plt.tight_layout()
plt.show()




