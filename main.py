import gymnasium as gym

env = gym.make("Taxi-v3", render_mode='human').env
state, _ = env.reset()

action = env.action_space.sample(env.action_mask(state))
next_state, reward, done, _, _ = env.step(action)

env.render()
print("Action Space{}".format(env.action_space))
print("State Space{}".format(env.observation_space))