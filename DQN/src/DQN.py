from __future__ import annotations
import torch as T
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class DeepQNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, lr):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, states):
        states = F.relu(self.layer1(states))
        states = F.relu(self.layer2(states))
        return self.layer3(states)
    
class ReplayMemory(object):
    def __init__(self, size):
        self.memory = deque([], maxlen=size)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

