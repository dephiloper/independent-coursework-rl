#! /usr/bin/env python3
import random

from gym_teeworlds import Action, TeeworldsMultiEnv, TeeworldsEnv

action = Action()
action.direction = 1
i = 0
# single_env = TeeworldsEnv()

# two multi envs with 2 players each, all controllable
multi_envs = [TeeworldsMultiEnv(n=2, teeworlds_srv_port="8303"), TeeworldsMultiEnv(n=2, teeworlds_srv_port="8304")]

while True:
    action.mouse_x = random.randrange(-200, 200)
    action.mouse_y = random.randrange(-200, 200)
    action.direction *= -1 if i % 30 == 0 else 1
    action.jump = 1 if i % 20 else 0
    action.hook = 1 if i % 40 else 0
    action.shoot = 1 if i % 50 else 0
    i += 1
    observation, reward, done, info = multi_envs[i % len(multi_envs)].step(action)
    # multi_envs[0].step_by_id(action, 0) # use this if you only want to perform any action with one client
