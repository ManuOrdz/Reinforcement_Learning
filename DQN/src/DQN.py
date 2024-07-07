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

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, n_actions)
    
    def forward(self, states):
        states = F.relu(self.layer1(states))
        states = F.relu(self.layer2(states))
        return self.layer3(states)
    
class ReplayMemory(object):
    def __init__(self, size, device="cpu"):
        self.memory = deque([], maxlen=size)
        self.device = device
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

