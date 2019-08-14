from tqdm import tqdm
from queue import Queue
from typing import List

import cv2
import numpy as np
from threading import Thread, Lock, Event

import torch

from dqn_teeworlds import Net, Experience, ExperienceBuffer, REPLAY_SIZE, actions, DEVICE, EPSILON_START,\
    LEARNING_RATE, REPLAY_START_SIZE, SYNC_TARGET_FRAMES, BATCH_SIZE, calc_loss
from gym_teeworlds import NUMBER_OF_IMAGES, start_mon, Action, TeeworldsEnv, OBSERVATION_SPACE

NUM_WORKERS = 1
NUM_TRAININGS_PER_EPOCH = 100
COLLECT_EXPERIENCE_SIZE = 200


class Worker(Thread):
    def __init__(self, experience_queue: Queue, net: Net, actions: List, device: str = 'cpu'):
        Thread.__init__(self)

        self.experience_queue = experience_queue
        self.net = net
        self.epsilon = 1.0
        self.actions = actions
        self.device = device

        self.env = TeeworldsEnv(mon=start_mon)
        self.state = self.env.reset()

        self._running = Event()

    def start_collecting_experience(self):
        self.env.reset()
        self._running.set()

    def stop_collecting_experience(self):
        self._running.clear()

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def run(self) -> None:
        while True:
            self._running.wait()

            # with probability epsilon take random action (explore)
            if np.random.random() < self.epsilon:
                index = np.random.randint(len(self.actions))  # np.random.choice(actions)
            else:  # otherwise use the past model to obtain the q-values for all possible actions, choose the best
                state_a = np.array(
                    self.state,
                    copy=False,
                    dtype=np.float32
                ).reshape(
                    (1, NUMBER_OF_IMAGES, start_mon['width'], start_mon['height'])
                )
                state_v = torch.tensor(state_a, dtype=torch.float32).to(self.device)
                if not self._running.is_set():
                    continue  # this could break, because concurrent modification of self.net

                q_values_v = self.net(state_v)  # calculate q values
                index = torch.argmax(q_values_v)  # get index of value with best outcome

            action = self.actions[index]  # extracting action from index ([0,-1], [0,0], [0,1]) <- (0, 1, 2)

            # do step in the environment
            new_state, reward, is_done, _ = self.env.step(Action.from_list(action))

            assert(new_state is not None)

            # store experience in exp_buffer
            exp = Experience(self.state, index, reward, is_done, new_state)
            self.experience_queue.put(exp)

            self.state = new_state

            # end of episode situation
            if is_done:
                self.env.reset()


def main():
    workers = []
    assert(SYNC_TARGET_FRAMES % COLLECT_EXPERIENCE_SIZE == 0)
    experience_queue = Queue()
    observation_size = OBSERVATION_SPACE.shape
    epsilon = EPSILON_START

    net = Net(observation_size, n_actions=len(actions)).to(DEVICE)
    target_net = Net(observation_size, n_actions=len(actions)).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for _ in range(NUM_WORKERS):
        worker = Worker(experience_queue, net, actions, DEVICE)
        workers.append(worker)

    experience_buffer = ExperienceBuffer(capacity=REPLAY_SIZE)

    for worker in workers:
        worker.start()
        worker.start_collecting_experience()

    frame_idx = 0

    while True:
        for _ in tqdm(range(COLLECT_EXPERIENCE_SIZE)):
            experience_buffer.append(experience_queue.get())
            frame_idx += 1

        if len(experience_buffer) < 400:  # check if buffer is large enough for training
            continue
        elif len(experience_buffer) == REPLAY_START_SIZE:
            print('start training')

        if frame_idx % SYNC_TARGET_FRAMES == 0:  # sync nets (copy weights)
            target_net.load_state_dict(net.state_dict())

        for worker in workers:
            worker.stop_collecting_experience()

        # learning
        for _ in tqdm(range(NUM_TRAININGS_PER_EPOCH)):
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
