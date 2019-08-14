#! /usr/bin/env python3
import random

from gym_teeworlds import Action, TeeworldsEnv
from utils import Monitor

action = Action()
action.direction = 1
i = 0
single_env = TeeworldsEnv(monitor=Monitor(800, 40, 840, 840), server_tick_speed=50, is_human=True, game_information_port=5005)

# two multi envs with 2 players each, all controllable
# multi_envs = [TeeworldsMultiEnv(n=2, teeworlds_srv_port="8303")]  # , TeeworldsMultiEnv(n=2, teeworlds_srv_port="8304")]

while True:
    if i % 2000 == 0:
        single_env.reset()
    action.mouse_x = random.randrange(-200, 200)
    action.mouse_y = random.randrange(-200, 200)
    action.direction *= -1 if i % 30 == 0 else 1
    action.jump = 1 if i % 20 else 0
    action.hook = 1 if i % 40 else 0
    action.shoot = 1 if i % 50 else 0
    i += 1
    # observation, reward, done, game_information = single_env.step(action)
    observation, reward, done, info = single_env.step(action)
    if reward > 0:
        print("reward:" + str(reward))
    # cv2.imshow("x", observation[0])
    # cv2.waitKey()
    # multi_envs[0].step_by_id(action, 0) # use this if you only want to perform any action with one client
