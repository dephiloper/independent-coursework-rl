import roboschool
import gym
import torch
import numpy as np

from roboschool_pong_deep_q_learning import Net, HIDDEN_SIZE, actions


ENV_NAME = 'RoboschoolPong-v1'
MODEL_NAME = 'RoboschoolPong-v1-best.dat'


if __name__ == '__main__':
    env = gym.make(ENV_NAME)
    state = env.reset()

    net = Net(env.observation_space.shape[0], HIDDEN_SIZE, len(actions))
    net.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

    while True:
        env.render()
        state_v = torch.tensor(np.array([state], copy=False, dtype=np.float32))

        q_vals = net(state_v).data.numpy()[0]
        action_index = np.argmax(q_vals)
        action = actions[action_index]
        state, reward, done, _ = env.step(action)
