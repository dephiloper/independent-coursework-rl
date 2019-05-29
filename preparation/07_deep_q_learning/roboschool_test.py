import datetime

import roboschool
import gym
import torch
import numpy as np

from roboschool_pong_deep_q_learning import Net, HIDDEN_SIZE, MONITOR_DIRECTORY, actions


ENV_NAME = 'RoboschoolPong-v1'
MODEL_NAME = 'models/LR5-10L2x64.dat'


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    state = env.reset()

    video_recorder = gym.monitoring.video_recorder.VideoRecorder(
        env=env,
        base_path='vids/pong_{}'.format(datetime.datetime.now())
    )

    net = Net(env.observation_space.shape[0], HIDDEN_SIZE, len(actions))
    net.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

    for i in range(10000):
        env.render()
        state_v = torch.tensor(np.array([state], copy=False, dtype=np.float32))

        q_vals = net(state_v).data.numpy()[0]
        action_index = np.argmax(q_vals)
        action = actions[action_index]
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.env.close()
