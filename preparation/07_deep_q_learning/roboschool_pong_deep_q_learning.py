import time
from datetime import datetime

import gym
import numpy as np
import roboschool
import torch
from future.moves import collections
from torch import nn, optim

from tensorboardX import SummaryWriter

'''
DQN-Algorithm
1. Initialize parameters for Q(s,a) and Q'(s,a) with random weights, epsilon with 1.0 and create empty replay buffer
2. With probability epsilon, select random action a, otherwise a = argmax_a Q(s,a)
3. Execute action a in environment and observe reward r + next state s'
4. Store transition in replay buffer (s,a,r,s') <- 10000
5. Sample random batch from replay buffer (32)
6. For every transition in replay buffer calculate y=r when episode has ended, otherwise y=GAMMA*max_a'(sum(Q'(s',a'))
7. Calculate loss L = (Q_s,a - y)^2
8. Update Q(s,a) using SGD algorithm by minimizing the loss
9. Every N (1000) Steps copy weights from Q to Q'_t
10 Repeat from step 2 until converged
'''

ENV_NAME = "RoboschoolPong-v1"
MEAN_REWARD_BOUND = 10      # initial value:    10 (randomly guessed) <- this needs to be checked
HIDDEN_SIZE = 64            # initial value:   128 (randomly guessed)
GAMMA = 0.99                # initial value:    99 (bellman equation, used for conv's eventually 0.9 would fit better)
BATCH_SIZE = 32             # initial value:    32 (sample size of experiences from replay buffer)
REPLAY_START_SIZE = 10000   # initial value: 10000 (min amount of experiences in replay buffer)
REPLAY_SIZE = 10000         # initial value: 10000 (max capacity of replay buffer)
LEARNING_RATE = 1e-5        # initial value:  1e-4 (also quite low eventually using default 1e-3)
SYNC_TARGET_FRAMES = 1000   # initial value   1000 (how frequently we sync target net with net)

# used for epsilon decay schedule
# -> starting at epsilon 1.0: only explore
# -> during first 100.000 steps epsilon is decayed linear to 0.02: explore 2% of the time, otherwise exploit
EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

DEVICE = "cpu"
MONITOR_DIRECTORY = './vids'
VIDEO_INTERVAL = 200

# stay, left, right, back, forward, left-down, right-down, left-forward, right-forward 
actions = np.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [1, -1], [-1, 1], [1, 1]])

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.net(x)


# keep the last transitions obtained from the environment (s, a, r, done_flag, s')
# every time we do a step we push the transition into the buffer
# keeping only a fixed number of steps
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    # for training we randomly sample the batch of transitions from the replay buffer
    # this allows to break the correlation between subsequent steps in the environment
    def sample(self, batch_size):
        # controls whether the sample is returned to the sample pool, for unique samples this should be false
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states, dtype=np.float32), \
               np.array(actions, dtype=np.int64), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states, dtype=np.float32)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    # perform a step in the environment and store the result in the replay buffer
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # with probability epsilon take random action (explore)
        if np.random.random() < epsilon:
            index = np.random.randint(0, len(actions))
        else:  # otherwise use use the past model to obtain the q-values for all possible actions, choose the best
            state_a = np.array(self.state, copy=False, dtype=np.float32)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)  # calculate q values
            index = torch.argmax(q_vals_v)  # get index of value with best outcome

        action = actions[index]  # extracting action from index ([0,-1], [0,0], [0,1]) <- (0, 1, 2)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        # store experience in exp_buffer
        exp = Experience(self.state, index, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state

        # end of episode situation
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward  # returns total reward of episode if we reached the end, otherwise None


# TODO needs to be discussed!
# performs the loss calculation mentioned in 6 and 7
# equations:
#   https://bit.ly/2I7iBqa <- steps which aren't end of episode
#   https://bit.ly/2HFsOec <- final steps
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch  # unpack the sample

    # wrap numpy data in torch tensors <- execution on gpu fast like sonic
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # extract q-values for taken actions using the gather function
    # gather explained: https://stackoverflow.com/a/54706716/10547035 or page 144
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    # apply target network to next state observations, and calculate the maximum q-value along the same action dim 1
    # max returns both max value and indices of those values
    next_state_values = tgt_net(next_states_v).max(1)[0]

    # this part is mentioned as very important!
    # if transition in the batch is from the last step in the episode, then our value of the action does not
    # have a discounted reward of the next state, as there is no next state to gather reward from
    # without this training will not converge!
    next_state_values[done_mask] = 0.0

    # detach() returns tensor w/o connection to its calculation history
    # detach value from computation graph, which prevents gradients of flowing into target network
    # when not performed backprop of the loss will affect both nets
    next_state_values = next_state_values.detach()

    # calculate the bellman approximation value
    expected_state_action_values = next_state_values * GAMMA + rewards_v

    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    env = gym.make(ENV_NAME)

    # video_env = gym.make(ENV_NAME)
    # video_env = gym.wrappers.Monitor(video_env, MONITOR_DIRECTORY, video_callable=lambda _: True, force=True)

    # video_env.reset()

    env.reset()

    observation_size = env.observation_space.shape[0]
    action_size = len(actions)

    net = Net(observation_size, HIDDEN_SIZE, action_size).to(DEVICE)
    target_net = Net(observation_size, HIDDEN_SIZE, action_size).to(DEVICE)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    total_loss = [0]
    
    # counter of frames to track current speed
    frame_idx = 0
    ts_frame = 0
    game_idx = 0
    ts = time.time()
    best_mean_reward = None  # every time mean reward beats record, we'll save model in file

    writer = SummaryWriter()
    
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)  # decrease epsilon

        reward = agent.play_step(net, epsilon, device=DEVICE)  # single step in env
        if reward is not None:
            game_idx += 1
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()

            mean_reward = np.mean(total_rewards[-100:])
            mean_loss = np.mean(total_loss)
            print("%d: done %d games, mean reward %.3f, eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), mean_reward, epsilon, speed))

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            writer.add_scalar("loss", mean_loss, frame_idx)
            total_loss = [0]

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), ENV_NAME + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:  # when boundary reached --> game finished
                print("Solved in %d frames!" % frame_idx)
                break

            # if game_idx % VIDEO_INTERVAL == 0:
                # video_env.reset()
                # agent.env = video_env
            # else:
                # agent.env = env

        if len(buffer) < REPLAY_START_SIZE:  # check if buffer is large enough for training
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:  # sync nets (copy weights)
            target_net.load_state_dict(net.state_dict())

        # learning
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)

        # perform optimization by minimizing the loss
        loss_t = calc_loss(batch, net, target_net, device=DEVICE)
        total_loss.append(loss_t.item())
        loss_t.backward()
        optimizer.step()

    writer.close()
    # video_env.env.close()
