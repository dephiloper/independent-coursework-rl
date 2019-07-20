#! /usr/bin/env python3
import random

from gym_teeworlds import Action, TeeworldsEnv

action = Action()
action.direction = 1
i = 0
env = TeeworldsEnv("5555", "5556")

while True:
    action.mouse_x = random.randrange(-200, 200)
    action.mouse_y = random.randrange(-200, 200)
    action.direction *= -1 if i % 30 == 0 else 1
    action.jump = 1 if i % 20 else 0
    action.hook = 1 if i % 40 else 0
    action.shoot = 1 if i % 50 else 0
    i += 1
    obs = env.step(action)
