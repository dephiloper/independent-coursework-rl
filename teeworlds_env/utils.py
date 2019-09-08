import queue
import random
import string

import numpy as np
import yaml
from future.moves import collections
from screeninfo import screeninfo

Experience = collections.namedtuple(
    'Experience',
    field_names=['state', 'action', 'reward', 'done', 'new_state', 'worker_index']
)

# move left, stay, move top, move left + jump, stay + jump, move top + jump
ACTIONS = [[-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
ACTION_LABELS = ['left', 'stay', 'right', 'jump left', 'jump', 'jump right']


def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config


class Monitor:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    def to_dict(self):
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }

    def copy(self):
        return Monitor(self.left, self.top, self.width, self.height)


def mon_iterator(n, width, height, top_spacing=0, x_padding=0, y_padding=0):
    """
    Yields n Monitors with the given width and height

    :param n: The number of Monitors to create
    :param width: The width of every Monitor
    :param height: The height of every Monitor
    :param top_spacing: The spacing over the monitors
    :param x_padding: x Padding between monitors
    :param y_padding: y Padding between monitors
    :return: An iterator over Monitors that do no overlap and have the given dimensions
    """
    screen_width = screeninfo.get_monitors()[0].width
    screen_height = screeninfo.get_monitors()[0].height

    x = 0
    y = top_spacing

    for i in range(n):
        yield Monitor(x, y, width, height)

        x += width + x_padding

        if x + width > screen_width:
            x = 0
            y += height + y_padding

        if y + height > screen_height:
            raise ToManyMonitorsError(
                'Could not create {} monitors of size {}, because insufficient screen space'.format(n, (width, height))
            )


def get_all_from_queue(q: queue.Queue) -> list:
    result_list = []
    while True:
        try:
            result_list.append(q.get_nowait())
        except queue.Empty:
            break
    return result_list


class ToManyMonitorsError(Exception):
    pass


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
        states, actions, rewards, dones, next_states, worker_index = zip(*[self.buffer[idx] for idx in indices])
        try:
            return np.array(states, dtype=np.float32), \
                   np.array(actions, dtype=np.int64), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(dones, dtype=np.uint8), \
                   np.array(next_states, dtype=np.float32)
        except ValueError as e:
            print(str(e))


class PriorityExperienceBuffer:
    def __init__(self, capacity, prob_alpha=0.6):
        self.buffer = []  # circular buffer
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:  # buffer has not reached the maximum capacity -> just append
            self.buffer.append(experience)
        else:  # buffer is already full -> override oldest transition from left to right until ends reached -> restart
            self.buffer[self.pos] = experience

        # when adding new experiences to the buffer those get the maximum priority, to make sure they be sampled soon
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:  # when buffer is full take all priorities
            priorities = self.priorities
        else:  # when buffer not full yet, take everything until last inserted position
            priorities = self.priorities[:self.pos]

        # convert priorities to probabilities using alpha hyper-parameter
        probabilities = priorities ** self.prob_alpha
        probabilities /= probabilities.sum()

        # sample our buffer using the probabilities determined above
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        states, actions, rewards, dones, next_states, worker_index = zip(*[self.buffer[idx] for idx in indices])

        samples = np.array(states, dtype=np.float32), \
            np.array(actions, dtype=np.int64), \
            np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), \
            np.array(next_states, dtype=np.float32)

        # calculate weights for samples in the batch (the value for each sample is defined as w_i = (N * P(i))^(-beta)
        # beta is a hyper-parameter between 0 and 1, for good convergence beta starting at 0.4 slowly increasing to 1.0
        # see age 185
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # indices for batch samples are required to update priorities for sampled items
        return samples, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        """
        allows to update new priorities for a processed batch
        """
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority


def random_id(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])
