#Imports and Environment Setup
from __future__ import annotations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


from Parameters import QParameters  
from Agent import QLearning

from utils import plot_performance, moving_average
# Create the enviroment

env = gym.make('MountainCar-v0', render_mode='rgb_array', max_episode_steps=400)

action_space = env.action_space
observation_space = env.observation_space

show_info = False
if show_info:
        #Observing the enviroment
    observation, info = env.reset()
    print(f'Observation:{observation}')
    print(f'Information:{info}')

    #sample a random action from all valid actions
    print(f'Action space: {action_space}')
    print(f'Observation space: {observation_space}')
    action = env.action_space.sample()
    #execute the action in our enviroment and receive infos from the environment

    observation, reward, terminated, truncated, info = env.step(action)
    print("New state obtained from action:",action)
    print(f'Observation:{observation} \nReward:{reward}\nTerminated:{terminated}')

param = QParameters(n_episodes=1000, learning_rate=0.05, final_epsilon=0.1, gamma=0.99)
##hiperparameters
learning_rate = param.learning_rate
n_episodes = param.n_episodes
start_epsilon = param.start_epsilon
epsilon_decay = param.epsilon_decay
final_epsilon = param.final_epsilon

env = gym.wrappers.RecordVideo (env=env, 
                               video_folder='QLearning/tmp',
                               name_prefix='test-video',
                               episode_trigger=lambda x: x%1000 == 0)

env = gym.wrappers.RecordEpisodeStatistics(env=env, deque_size=n_episodes)
agent = QLearning(
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
    q_size=(20,20),
    observation_space=observation_space,
    action_space = action_space,
)
# Training agent
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        #update the agent
        agent.update(obs, action, reward, next_obs)
        #update if the environment is done and the current state
        
        done = terminated or truncated
        obs = next_obs
    agent.decay_epsilon()


# Visualize training metrics
training_error = np.array(agent.training_error)
training_error_average = []
w = 1000
len_window = int(len(training_error) / w)
for i in range(w):
    training_error_average.append(training_error[(i)*len_window:(i+1)*len_window].mean())

reward_ = np.array(env.return_queue).flatten()
lenght_ = np.array(env.length_queue).flatten()

training_error_moving_average = moving_average(training_error_average, 50)
reward_moving_average = moving_average(reward_, 50)
length_moving_average = moving_average(lenght_, 50)

plot_performance(reward_moving_average, length_moving_average, training_error_moving_average)

