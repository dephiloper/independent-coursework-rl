import random

class Agent:
	def __init__(self):
		self.total_reward = 0.0

	def step(self, env):
		current_obs = env.get_observation()
		actions = env.get_actions()
		reward = env.action(random.choice(actions))
		self.total_reward += reward
