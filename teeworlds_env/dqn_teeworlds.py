import time

import gym
import numpy as np
import roboschool
import torch
from future.moves import collections
from torch import nn, optim

from tensorboardX import SummaryWriter

from gym_teeworlds import TeeworldsCoopEnv, Action, NUMBER_OF_IMAGES, start_mon, TeeworldsMultiEnv

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

MODEL_NAME = 'L4'
MEAN_REWARD_BOUND = 10  # initial value:    10 (randomly guessed) <- this needs to be checked
HIDDEN_SIZE = 64  # initial value:   128 (randomly guessed)
GAMMA = 0.99  # initial value:    99 (bellman equation, used for conv's eventually 0.9 would fit better)
BATCH_SIZE = 256  # initial value:    32 (sample size of experiences from replay buffer)
REPLAY_START_SIZE = 10000  # initial value: 10000 (min amount of experiences in replay buffer)
REPLAY_SIZE = 10000  # initial value: 10000 (max capacity of replay buffer)
LEARNING_RATE = 1e-4  # initial value:  1e-4 (also quite low eventually using default 1e-3)
SYNC_TARGET_FRAMES = 10000  # initial value   1000 (how frequently we sync target net with net)

# used for epsilon decay schedule
# -> starting at epsilon 1.0: only explore
# -> during first 100.000 steps epsilon is decayed linear to 0.02: explore 2% of the time, otherwise exploit
EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

DEVICE = "cpu"
MONITOR_DIRECTORY = './vids'
VIDEO_INTERVAL = 12

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

# move left, stay, move top, move left + jump, stay + jump, move top + jump
actions = [[-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

        # in_channel of the first conv layer is the number color's / color depth of the image and represents later
        # the amount of channels, which means output filter of the first conv needs to be input of the second and so on
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


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
        try:
            return np.array(states, dtype=np.float32), \
                   np.array(actions, dtype=np.int64), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(dones, dtype=np.uint8), \
                   np.array(next_states, dtype=np.float32)
        except ValueError as e:
            print(str(e))


class Agent:
    def __init__(self, env, exp_buffer):
        self.state = NotImplementedError()
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        if self.state is None:
            print('None state')
        self.total_reward = 0.0

    # perform a step in the environment and store the result in the replay buffer
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        # with probability epsilon take random action (explore)
        if np.random.random() < epsilon:
            index = np.random.randint(len(actions))  # np.random.choice(actions)
        else:  # otherwise use the past model to obtain the q-values for all possible actions, choose the best
            try:
                state_a = np.array(
                    self.state,
                    copy=False,
                    dtype=np.float32
                ).reshape(
                    (1, NUMBER_OF_IMAGES, start_mon['width'], start_mon['height'])
                )
            except ValueError as e:
                print('error: {}'.format(e))
                print('state: {}'.format(self.state))
                # print('state.shape: {}'.format(self.state.shape))
                print('')
            state_v = torch.tensor(state_a, dtype=torch.float32).to(device)
            q_vals_v = net(state_v)  # calculate q values
            index = torch.argmax(q_vals_v)  # get index of value with best outcome

        action = actions[index]  # extracting action from index ([0,-1], [0,0], [0,1]) <- (0, 1, 2)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(Action.from_list(action))
        self.total_reward += reward

        # store experience in exp_buffer
        exp = Experience(self.state, index, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        if new_state is None:
            print('None state')
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
# noinspection PyCallingNonCallable,PyUnresolvedReferences
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch  # unpack the sample

    # wrap numpy data in torch tensors <- execution on gpu fast like sonic
    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
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


def x(episode_id):
    if episode_id % VIDEO_INTERVAL == 0:
        print('capturing episode: {}'.format(episode_id))
        return True
    return False


if __name__ == "__main__":
    env = TeeworldsMultiEnv(4)
    # env.reset()

    observation_size = env.observation_space.shape

    net = Net(observation_size, n_actions=len(actions)).to(DEVICE)
    target_net = Net(observation_size, n_actions=len(actions)).to(DEVICE)

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
                torch.save(net.state_dict(), "teeworlds-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f, model saved" % (best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
            if mean_reward > MEAN_REWARD_BOUND:  # when boundary reached --> game finished
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:  # check if buffer is large enough for training
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:  # sync nets (copy weights)
            target_net.load_state_dict(net.state_dict())

        # if (frame_idx % (1000 * VIDEO_INTERVAL)) == 0:
        #    torch.save(net.state_dict(), 'live_models/{}_frame{}.dat'.format(MODEL_NAME, frame_idx))

        # learning
        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)

        # perform optimization by minimizing the loss
        loss_t = calc_loss(batch, net, target_net, device=DEVICE)
        total_loss.append(loss_t.item())
        loss_t.backward()
        optimizer.step()

    writer.close()
