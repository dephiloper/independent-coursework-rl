import random

class Environment:
	def __init__(self):
		self.steps_left = 10

	def get_observation(self):
		return [0.0, 0.0, 0.0]

	def get_actions(self):
		return [0, 1]

	def is_done(self):
		return self.steps_left == 0

	def action(self, action):
		if self.is_done():
			raise Exception("Game is over")
		self.steps_left -= 1
		return random.random() # returns the reward
