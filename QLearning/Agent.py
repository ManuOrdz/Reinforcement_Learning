from __future__ import annotations

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

class Plot():
    def __init__(self, **kwargs):
        self.fig, self.axs = plt.subplots(ncols = 3, figsize=(12,5), tight_layout=True)
        pass
    
    def plot(self, reward_moving_average, length_moving_average, training_error_moving_average):
        self.axs[0].set_title("Episode rewards")
        self.axs[0].plot(range(len(reward_moving_average)),reward_moving_average)
        
        self.axs[1].set_title("Lenght Episode")
        self.axs[1].plot(range(len(length_moving_average)),length_moving_average)
        
        self.axs[2].set_title("Training Error")
        self.axs[2].plot(range(len(training_error_moving_average)),training_error_moving_average)

class Metrics:
    def __init__(
        self,
        training_error: np.array | list,
        reward_episode: np.array | list,
        length_episode: np.array | list,
        rolling_length: int = 100,
    ):
        self.training_error = np.array(training_error)
        self.reward_episode = np.array(reward_episode).flatten()
        self.length_episode = np.array(length_episode).flatten()
        self.rolling_length = rolling_length

    def moving_average(self, 
                       kind: str = 'training') -> np.array:
        """Calcule moving average for metrics smother

        Args:
            rolling_length (int): rolling lenght
            kind (str, optional): variable to applay Defaults to 'training'.

        Returns:
            np.array: moving average array
        """
                    
        if kind == 'training':
            a = self.training_error
            mode = 'full'
        elif kind == 'reward':
            a = self.reward_episode
            mode = 'valid'
        elif kind == 'length':
            a = self.length_episode
            mode = 'same'
            
        moving_average = (
            np.convolve(
                a=a, v=np.ones(self.rolling_length)
            ) / self.rolling_length
        )
        return moving_average

class QLearning:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        observation_space: gym.spaces.box.Box,
        action_space: gym.spaces.space.discrete.Discrete,
        q_size: tuple = (10,10),
        gamma: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an zero array
        of state-action table (q_table), a learning rate an epsilon.
        
        Args:
            learning_rate (float): The learning rate
            initial_epsilon: The inital epsilon value
            epsilon_decay: The decay for epsilon
            observation_space: The observation space
            action_space: The action space
            final_epsilon: The final epsilon value
            size_table: Size of the Q-table (observation_size, actions)
            gamma: The discount factor for computing the Q-value
        """
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.observation_space = observation_space
        self.action_space = action_space
        self.q_size = q_size   
        self.gamma = gamma
        self.reset_qtable()
        self.training_error = []
    
    def map_obs_to_stat(self, obs_value) -> tuple:
        """ Discretize observation value and map to a state value

        Args:
            obs_value (float): observation_value
        """        
        obs_min = self.observation_space.low
        obs_max = self.observation_space.high
        aux = (obs_value - obs_min) / (obs_max - obs_min)
        aux = aux*self.q_size
        return tuple(aux.astype('uint8'))
    
    def get_action(self, obs: tuple[int,int]) -> int:
        """Returns the best action with probability (1-epsilon)
        otherwise a random action with probability epsilon to ensure exploration

        Args:
            state (tuple[int,int]): observation

        Returns:
            int: action
        """
        state = self.map_obs_to_stat(obs)
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return np.argmax(self.q_table[state])
        
    def update(
        self,
        obs: tuple[int, int],
        action: int,
        reward: float,
        next_obs: tuple[int, int],
    ):
        """Updates the Q-value of an action
        Update Q(s,a):= Q(s,a) + lr[R(s,a) + gamma * max(Q(s',a')-Q(s,a))]

        Args:
            state (tuple[int, int]): actual state
            action (int): action
            reward (float): reward
            next_state (tuple[int, int]): next state
        """
        state = self.map_obs_to_stat(obs)
        next_state = self.map_obs_to_stat(next_obs)
        
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
        """ Aplay de epsilon decay
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
    def reset_qtable(self):
        """Reset the Q-table = 0"""
        act_size = self.action_space.n
        q_size = tuple(size for size in (self.q_size+(act_size,)))
        self.q_table = np.zeros(shape=q_size)
        
    def metrics(
        self, 
        training_error: np.array | list, 
        reward_episode: np.array | list, 
        length_episode: np.array | list,
        rolling_length: int = 100,
        plot: bool = False,
        **kwargs,
        ):
        """ Impement metric and plot objets to visualize performance of the Qlearning algorithm

        Args:
            training_error (np.array | list): training array 
            reward_episode (np.array | list): reward array
            length_episode (np.array | list): lenght array
            rolling_length (int, optional): moving average lenght. Defaults to 100.
            plot (bool, optional): Define if plot or not. Defaults to False.
        """
        metrics = Metrics(training_error, reward_episode, length_episode, rolling_length) 
               
        self.training_error_moving_average = metrics.moving_average('training')
        self.reward_moving_average = metrics.moving_average('reward')
        self.length_moving_average = metrics.moving_average('length')
        
        if plot:
            p = Plot()
            p.plot(self.reward_moving_average,
                   self.length_moving_average,
                   self.training_error_moving_average)

