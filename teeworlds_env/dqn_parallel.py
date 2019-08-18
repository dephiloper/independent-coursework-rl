import time

from tensorboardX import SummaryWriter
from tqdm import tqdm
from typing import List

import numpy as np
from torch.multiprocessing import Process, Queue, Value
from queue import Empty

import torch

from dqn_teeworlds import Net, Experience, ExperienceBuffer, REPLAY_SIZE, actions, DEVICE, EPSILON_START,\
    LEARNING_RATE, REPLAY_START_SIZE, SYNC_TARGET_FRAMES, BATCH_SIZE, calc_loss
from gym_teeworlds import NUMBER_OF_IMAGES, Action, OBSERVATION_SPACE, teeworlds_env_settings_iterator, \
    TeeworldsEnvSettings

NUM_WORKERS = 8
NUM_TRAININGS_PER_EPOCH = 10
COLLECT_EXPERIENCE_SIZE = 2000
SERVER_TICK_SPEED = 50
MONITOR_WIDTH = 84
MONITOR_HEIGHT = 84
EPSILON_DECAY = 0.01
MIN_EPSILON = 0.02


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
            if np.random.random() < self.epsilon:
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

            assert(new_state is not None)

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
    torch.multiprocessing.set_start_method('spawn')
    assert(SYNC_TARGET_FRAMES % COLLECT_EXPERIENCE_SIZE == 0)

    workers = []
    experience_queue = Queue()
    stats_queue = Queue()

    observation_size = OBSERVATION_SPACE.shape
    epsilon = Value('d', EPSILON_START)

    net = Net(observation_size, n_actions=len(actions)).to(DEVICE)
    target_net = Net(observation_size, n_actions=len(actions)).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for env_setting in teeworlds_env_settings_iterator(
            NUM_WORKERS,
            MONITOR_WIDTH,
            MONITOR_HEIGHT,
            top_spacing=40,
            server_tick_speed=SERVER_TICK_SPEED
    ):
        worker = Worker(env_setting, experience_queue, stats_queue, net, epsilon, actions, DEVICE)
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

    while True:
        for _ in tqdm(range(COLLECT_EXPERIENCE_SIZE), desc='collecting data: '):
            experience_buffer.append(experience_queue.get())
            frame_idx += 1

            while True:
                try:
                    game_stat = stats_queue.get_nowait()
                    game_stats.append(game_stat)
                    writer.add_scalar('reward', game_stat.reward, frame_idx)

                    reward_100 = 0
                    for gs in game_stats[-100:]:
                        reward_100 += gs.reward
                    reward_100 /= len(game_stats[-100:])
                    writer.add_scalar('reward_100', reward_100, frame_idx)

                    writer.add_scalar('epsilon', epsilon.value, frame_idx)
                except Empty:
                    break

        if len(experience_buffer) < REPLAY_START_SIZE:  # check if buffer is large enough for training
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:  # sync nets (copy weights)
            target_net.load_state_dict(net.state_dict())

        for worker in workers:
            worker.stop_collecting_experience()

        epsilon.value = max(MIN_EPSILON, epsilon.value - EPSILON_DECAY)

        # for index, experience in enumerate(experience_buffer.buffer):
        #     cv2.imshow('frame{}'.format(index), experience.state[0])
        #     if cv2.waitKey() == 27:
        #         break
        # cv2.destroyAllWindows()

        # learning
        for _ in tqdm(range(NUM_TRAININGS_PER_EPOCH), desc='training: '):
            optimizer.zero_grad()
            batch = experience_buffer.sample(BATCH_SIZE)

            # perform optimization by minimizing the loss
            loss_t = calc_loss(batch, net, target_net, device=DEVICE)
            # total_loss.append(loss_t.item())
            loss_t.backward()
            optimizer.step()

        for worker in workers:
            worker.start_collecting_experience()

        """
        if frame_idx == 250:
            for i, experience in enumerate(experience_buffer.buffer):
                cv2.imshow(str(i), experience.state[0])
                cv2.waitKey()
        """


if __name__ == '__main__':
    main()
