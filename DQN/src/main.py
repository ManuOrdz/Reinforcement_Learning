from __future__ import annotations
import numpy as np
from DQN import DQN, ReplayMemory, Transition, T, optim, nn
import gymnasium as gym
from itertools import count
from tqdm import tqdm

device = 'cpu'
env = gym.make('MountainCar-v0')
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

memory = ReplayMemory(10000, device=device)
optimizer = optim.AdamW(policy_net.parameters(),lr=1e-4, amsgrad=True)
BATCH_SIZE = 32
TAU = 0.005
GAMMA = 0.95
steps_done = 0

class Agent(object):
    def __init__(self, device = 'cpu'):
        self.device = device
    
    def state_to_tensor(self, state) -> T.tensor:
        return T.tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
    
    def select_epsilon_greedy_action(self, state, epsilon):
        result = np.random.uniform()
        if result < epsilon:
            return T.tensor([[env.action_space.sample()]], device=self.device, dtype=T.long)
        else:
            with T.no_grad():
                return policy_net(state).max(1).indices.view(1, 1)
        
    def optimize(self):
        if len(memory) < BATCH_SIZE:
            return 
        
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=T.bool)
        non_final_next_states = T.cat([s for s in batch.next_state 
                                       if s is not None])
        
        state_batch = T.cat(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat (batch.reward)
        state_action_values = policy_net(state_batch).gather(1, action_batch)
        next_state_values = T.zeros(BATCH_SIZE, device=self.device)
        with T.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()
        
agent = Agent()
episode_durations = []        

for i_episode in tqdm(range(10)):
    state, info = env.reset()
    state = agent.state_to_tensor(state)
    for t in count():
        action = agent.select_epsilon_greedy_action(state, 0.2)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = T.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state =  agent.state_to_tensor(observation)
            
        memory.push(state, action, reward, next_state)
        state = next_state
        
        agent.optimize()
        
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            episode_durations.append(t+1)
            break
        

print(episode_durations)
