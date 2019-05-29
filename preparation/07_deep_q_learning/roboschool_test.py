import datetime

import roboschool
import gym
import torch
import numpy as np

from roboschool_pong_deep_q_learning import Net, HIDDEN_SIZE, MONITOR_DIRECTORY, actions


ENV_NAME = 'RoboschoolPong-v1'
MODEL_NAME = 'models/V56_L3_LR-4.dat'


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    state = env.reset()

    net = Net(env.observation_space.shape[0], HIDDEN_SIZE, len(actions))
    net.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

    game_counter = 0
    reward_sum = 0

    while game_counter < 100:
        if game_counter % 10 == 0:
            pass

        if not env.render():
            break

        state_v = torch.tensor(np.array([state], copy=False, dtype=np.float32))

        q_vals = net(state_v).data.numpy()[0]
        action_index = np.argmax(q_vals)
        action = actions[action_index]
        state, reward, done, _ = env.step(action)

        if reward == 1.0 or reward == -1.0:
            print('game {} reward: {}'.format(game_counter, reward))
            game_counter += 1
            reward_sum += reward

    if game_counter != 0:
        print('average reward: {}'.format(reward_sum/game_counter))

    env.env.close()
