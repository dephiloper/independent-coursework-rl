#! /usr/bin/env python3
import random
import sys

from gym_teeworlds import Action, TeeworldsEnv
from utils import Monitor


def main():
    action = Action()
    action.direction = 1
    i = 0
    env = TeeworldsEnv(
        monitor=Monitor(600, 40, 960, 540),
        server_tick_speed=200,
        is_human=True,
        game_information_port=5005,
        episode_duration=20,
        map_name='level_2'
    )

    env.reset()

    while True:
        action.mouse_x = random.randrange(-200, 200)
        action.mouse_y = random.randrange(-200, 200)
        action.direction = 1
        action.jump = 0 if i % 20 else 0
        action.hook = 0 if i % 40 else 0
        action.shoot = 0 if i % 50 else 0
        i += 1
        observation, reward, done, info = env.step(action)

        if reward != 0:
            print(reward)
            sys.stdout.flush()

        if done:
            print("done")
            env.reset()
        # cv2.imshow("x", observation[0])
        # cv2.waitKey()
        # multi_envs[0].step_by_id(action, 0) # use this if you only want to perform any action with one client


if __name__ == '__main__':
    main()
