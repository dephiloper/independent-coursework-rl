import os, sys, subprocess
import numpy as np
import gym
import roboschool
import torch

from roboschool_pong_deep_q_learning import Net, HIDDEN_SIZE, MONITOR_DIRECTORY, actions

MODEL_NAME = 'V69_L4.dat'


def play(env, net):
    episode_n = 0
    while True:
        episode_n += 1
        state = env.reset()
        while True:
            state_v = torch.tensor(np.array([state], copy=False, dtype=np.float32))
            q_vals = net(state_v).data.numpy()[0]
            action_index = np.argmax(q_vals)
            obs, rew, done, info = env.step(actions[action_index])
            if done:
                break
        #  break


print("test")

if len(sys.argv) == 1:
    import roboschool.multiplayer

    print("sys.argv", sys.argv)

    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = roboschool.multiplayer.SharedMemoryServer(game, "pongdemo", want_test_window=True)
    for n in range(game.players_count):
        subprocess.Popen([sys.executable, sys.argv[0], "pongdemo", "%i" % n])
    gameserver.serve_forever()

else:
    print("sys.argv", sys.argv)

    player_n = int(sys.argv[2])

    env = gym.make("RoboschoolPong-v1")
    env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=player_n)

    if player_n == 0:
        net = Net(env.observation_space.shape[0], HIDDEN_SIZE, len(actions))
        net.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))
    else:
        net = Net(env.observation_space.shape[0], HIDDEN_SIZE, len(actions))
        net.load_state_dict(torch.load(MODEL_NAME, map_location=lambda storage, loc: storage))

    play(env, net)  # set video = player_n==0 to record video
