from __future__ import annotations

from collections import defaultdict

import gymnasium as gym

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

class QLearning:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        q_size: tuple = (10,10,3),
        gamma: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action table (q_table), a learning rate an epsilon.
        
        Args:
            env: The environment
            learning_rate (float): The learning rate
            initial_epsilon: The inital epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            size_table: Size of the Q-table (observation_size, actions)
            discount_factor: The discount factor for computing the Q-value
        """
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        
        self.q_size = q_size        
        self.gamma = gamma
        self.training_error = []
        self.reset_qtable()
    
    def map_obs_to_stat(self, obs_value: tuple, obs_max:tuple, obs_min: tuple) -> tuple:
        """ Discretize observation value

        Args:
            obs_value (float): observation_value
        """        
        aux = (obs_value - obs_min) / (obs_max - obs_min)
        aux = (aux[0]*self.q_size[0]).astype('uint8'), (aux[1]*self.q_size[1]).astype('uint8')
        return aux
    
    def get_action(self, action_space, state: tuple[int,int]) -> int:
        """
        Returns the best action with probability (1-epsilon)
        otherwise a random action with probability epsilon to ensure exploration
        """
        
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return np.argmax(self.q_table[state])
        
    def update(
        self,
        state: tuple[int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int],
    ):
        """Updates the Q-value of an action
        Update Q(s,a):= Q(s,a) + lr[R(s,a) + gamma * max(Q(s',a')-Q(s,a))]

        Args:
            state (tuple[int, int]): actual state
            action (int): action
            reward (float): reward
            next_state (tuple[int, int]): next state
        """
        
        future_q_value = np.max(self.q_table[next_state])
        delta = (
            reward + self.gamma * future_q_value - self.q_table[state][action]
        )
        
        self.q_table[state][action] = (
            self.q_table[state][action] + self.lr * delta
        )
        self.rewards = []
        self.training_error.append(delta)
        
        
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
        
    def reset_qtable(self):
        """Reset the Q-table"""

        self.q_table = np.zeros(shape=(self.q_size))
