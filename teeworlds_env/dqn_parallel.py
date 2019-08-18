import time
from queue import Empty
from typing import List

import numpy as np
from torch.multiprocessing import Process, Queue, Value

import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

# training
from dqn_model import Net
from gym_teeworlds import teeworlds_env_settings_iterator, OBSERVATION_SPACE, TeeworldsEnvSettings, Action
from utils import ExperienceBuffer, ACTIONS, Experience

MODEL_NAME = "teeworlds-v0.1-"

# exp collecting
NUM_WORKERS = 1
COLLECT_EXPERIENCE_SIZE = 2000  # init: 2000 (amount of experiences to collect after each training step)
GAME_TICK_SPEED = 200  # default: 50 (game speed, when higher more screenshots needs to be captures)
MONITOR_WIDTH = 84  # init: 84 width of game screen
MONITOR_HEIGHT = 84  # init: 84 height of game screen (important for conv)

# training
REPLAY_START_SIZE = 4000  # init: 10000 (min amount of experiences in replay buffer before training starts)
REPLAY_SIZE = 10000  # init: 10000 (max capacity of replay buffer)
DEVICE = 'cpu'  # init: 'cpu'
BATCH_SIZE = 200  # init: 32 (sample size of experiences from replay buffer)
NUM_TRAININGS_PER_EPOCH = 50  # init: 50 (amount of BATCH_SIZE x NUM_TRAININGS_PER_EPOCH will be trained)
GAMMA = 0.99  # init: .99 (bellman equation)
MIN_EPSILON = 0.02  # init: 0.02
EPSILON_START = 1.0  # init: 1.0
EPSILON_DECAY = 0.01  # init: 0.01
LEARNING_RATE = 1e-4  # init: 1e-4 (also quite low eventually using default 1e-3)
SYNC_TARGET_FRAMES = 10000  # init: 1000 (how frequently we sync target net with net)

MEAN_REWARD_BOUND = 10  # init: 10 (randomly guessed) <- this needs to be checked


class GameStats:
    def __init__(self, reward):
        self.reward = reward


class Worker(Process):
    def __init__(
            self,
            env_settings: TeeworldsEnvSettings,
            experience_queue: Queue,
            stats_queue: Queue,
            net: Net,
            epsilon: Value,
            action_list: List,
            device: str = 'cpu'
    ):
        Process.__init__(self)

        self.experience_queue = experience_queue
        self.stats_queue = stats_queue
        self.net = net
        self.epsilon = epsilon
        self.actions = action_list
        self.device = device

        self.env_settings = env_settings
        self.total_reward = 0

        self._running_queue = Queue()
        self._running_queue.put(False)  # do not start immediately
        self.env = None
        self.state = None

    def start_collecting_experience(self):
        self._running_queue.put(True)

    def stop_collecting_experience(self):
        self._running_queue.put(False)

    def initialize_env(self):
        self.env = self.env_settings.create_env()
        self.state = self.env.reset()

    def _idle_for_running(self):
        try:
            token = self._running_queue.get_nowait()

            # if False was in queue
            if not token:
                self.state = self.env.reset()
                # wait for next true
                while not self._running_queue.get():
                    self.state = self.env.reset()
        except Empty:
            pass

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def run(self) -> None:
        self.initialize_env()

        while True:
            self._idle_for_running()

            # with probability epsilon take random action (explore)
            if np.random.random() < self.epsilon.value:
                index = np.random.randint(len(self.actions))  # np.random.choice(actions)
            else:  # otherwise use the past model to obtain the q-values for all possible actions, choose the best
                state_a = np.array(
                    self.state,
                    copy=False,
                    dtype=np.float32
                ).reshape(
                    (1, NUMBER_OF_IMAGES, self.env.monitor.width, self.env.monitor.height)
                )
                state_v = torch.tensor(state_a, dtype=torch.float32).to(self.device)

                q_values_v = self.net(state_v)  # calculate q values
                index = torch.argmax(q_values_v)  # get index of value with best outcome

            action = self.actions[index]  # extracting action from index ([0,-1], [0,0], [0,1]) <- (0, 1, 2)

            # do step in the environment
            new_state, reward, is_done, _ = self.env.step(Action.from_list(action))

            assert (new_state is not None)

            self.total_reward += reward

            # store experience in exp_buffer
            exp = Experience(self.state, index, reward, is_done, new_state)
            self.experience_queue.put(exp)

            self.state = new_state

            # end of episode situation
            if is_done:
                self.env.reset()
                self.stats_queue.put(GameStats(self.total_reward))
                self.total_reward = 0


def main():
    assert (SYNC_TARGET_FRAMES % COLLECT_EXPERIENCE_SIZE == 0)
    torch.multiprocessing.set_start_method('spawn')

    workers = []
    experience_queue = Queue()
    stats_queue = Queue()

    observation_size = OBSERVATION_SPACE.shape
    epsilon = Value('d', EPSILON_START)

    net = Net(observation_size, n_actions=len(ACTIONS)).to(DEVICE)
    target_net = Net(observation_size, n_actions=len(ACTIONS)).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for env_setting in teeworlds_env_settings_iterator(
            NUM_WORKERS,
            MONITOR_WIDTH,
            MONITOR_HEIGHT,
            top_spacing=40,
            server_tick_speed=GAME_TICK_SPEED,
    ):
        worker = Worker(env_setting, experience_queue, stats_queue, net, epsilon, ACTIONS, DEVICE)
        workers.append(worker)

    experience_buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

    for worker in workers:
        worker.start()
        time.sleep(2)

    time.sleep(4)

    for worker in workers:
        worker.start_collecting_experience()

    frame_idx = 0

    writer = SummaryWriter()
    game_stats = []

    max_mean_reward = 0
    finished_episodes = 0
    epoch = 0

    while True:
        # collect experience
        for _ in tqdm(range(COLLECT_EXPERIENCE_SIZE), desc='collecting data: '):
            experience_buffer.append(experience_queue.get())
            frame_idx += 1

            # gather stats for logging
            while True:
                try:
                    game_stat = stats_queue.get_nowait()
                    finished_episodes += 1
                except Empty:
                    break

                game_stats.append(game_stat)
                writer.add_scalar('reward', game_stat.reward, frame_idx)

                reward_100 = 0
                for stat in game_stats[-10:]:
                    reward_100 += stat.reward
                reward_100 /= len(game_stats[-10:])

                writer.add_scalar('reward_100', reward_100, frame_idx)
                writer.add_scalar('epsilon', epsilon.value, frame_idx)

        # check if buffer is large enough for training
        if len(experience_buffer) >= REPLAY_START_SIZE:

            # stop experience collection on all workers
            for worker in workers:
                worker.stop_collecting_experience()

            mean_reward = np.mean([stat.reward for stat in game_stats[-finished_episodes:]])
            if mean_reward > max_mean_reward:
                max_mean_reward = mean_reward
                torch.save(net.state_dict(), f"saves/{MODEL_NAME}_epoch-{epoch:04d}_rew-{max_mean_reward:08.0f}.dat")
            finished_episodes = 0

            # sync nets (copy weights)
            if frame_idx % SYNC_TARGET_FRAMES == 0:
                target_net.load_state_dict(net.state_dict())

            # decrease epsilon
            epsilon.value = max(MIN_EPSILON, epsilon.value - EPSILON_DECAY)

            # training
            for _ in tqdm(range(NUM_TRAININGS_PER_EPOCH), desc='training: '):
                optimizer.zero_grad()
                batch = experience_buffer.sample(BATCH_SIZE)

                # perform optimization by minimizing the loss
                loss_t = calc_loss(batch, net, target_net, device=DEVICE)
                # total_loss.append(loss_t.item())
                loss_t.backward()
                optimizer.step()

            epoch += 1

            # restart experience collection on all workers
            for worker in workers:
                worker.start_collecting_experience()


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
    return torch.nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == '__main__':
    main()
