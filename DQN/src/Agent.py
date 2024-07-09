import numpy as np
from DQN import DeepQNetwork, ReplayMemory, Transition, T, optim, nn

class Agent():
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        learning_rate: float = 1e-4,
        initial_epsilon: float = 1.,
        epsilon_decay: float = 1e-5,
        final_epsilon: float = 1e-2,
        gamma: float = 0.99,
    ):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        
        
        self.action_space = [i for i in range(self.n_actions)]
        
        self.policy_net = DeepQNetwork(self.n_observations, self.n_actions, self.lr)
        self.target_net = DeepQNetwork(self.n_observations, self.n_actions, self.lr)
        
    def choose_action(self, observation):
        if np.random.random() > self.epsilon :
            state = T.tensor(observation, dtype=T.float).to(self.policy_net.device)
            actions = self.policy_net.forward(state)
            return T.argmax(actions).item()
        else :
            return np.random.choice(self.action_space)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon-self.epsilon_decay)
    
agent = Agent(n_observations=2 , n_actions=3, initial_epsilon=0.1)





