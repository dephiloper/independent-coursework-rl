import queue
import random
import string

import numpy as np
from future.moves import collections
from screeninfo import screeninfo

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

# move left, stay, move top, move left + jump, stay + jump, move top + jump
ACTIONS = [[-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]


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


def mon_iterator(n, width, height, top_spacing=0):
    """
    Yields n Monitors with the given width and height

    :param n: The number of Monitors to create
    :param width: The width of every Monitor
    :param height: The height of every Monitor
    :param top_spacing: The spacing over the monitors
    :return: An iterator over Monitors that do no overlap and have the given dimensions
    """
    screen_width = screeninfo.get_monitors()[0].width
    screen_height = screeninfo.get_monitors()[0].height

    x = 0
    y = top_spacing

    for i in range(n):
        yield Monitor(x, y, width, height)

        x += width

        if x + width > screen_width:
            x = 0
            y += height

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
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        try:
            return np.array(states, dtype=np.float32), \
                   np.array(actions, dtype=np.int64), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(dones, dtype=np.uint8), \
                   np.array(next_states, dtype=np.float32)
        except ValueError as e:
            print(str(e))


def random_id(n):
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])
