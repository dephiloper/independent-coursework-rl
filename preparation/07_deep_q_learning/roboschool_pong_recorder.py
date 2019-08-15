#! /usr/bin/env python3
import subprocess
import sys
import time

import gym
import numpy as np
import roboschool
import glob
from tqdm import tqdm
import torch
from roboschool_pong_deep_q_learning import ENV_NAME, HIDDEN_SIZE, actions, Net


if __name__ == "__main__":
    if len(sys.argv) == 1:
        model_files = [f for f in glob.glob("models/tmp/*.dat")]

        for model_file in tqdm(model_files):
            p = subprocess.Popen([sys.executable, sys.argv[0], model_file])
            p.wait()

    if len(sys.argv) == 2:
        model_file = sys.argv[1]
        record_name = str(int(int(model_file.split("frame")[1].split(".dat")[0]) / 10000))

        env = gym.make(ENV_NAME)
        observation_size = env.observation_space.shape[0]
        action_size = len(actions)

        env = gym.wrappers.Monitor(env, directory="rec/" + record_name + "/", force=True)
        net = Net(observation_size, HIDDEN_SIZE, action_size)
        net.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        state = env.reset()
        while True:
            #env.render()
            state_v = torch.tensor(np.array([state], copy=False, dtype=np.float32))
            q_vals = net(state_v).data.numpy()[0]
            action_index = np.argmax(q_vals)
            state, rew, done, info = env.step(actions[action_index])
            if done:
                break
        env.env.close()
