import roboschool

import gym
import numpy as np


ENV_NAME = "RoboschoolPong-v1"

env = gym.make(ENV_NAME)
env.reset()

action = np.array([0, 0])

while True:
    env.step(action)
    env.render()
