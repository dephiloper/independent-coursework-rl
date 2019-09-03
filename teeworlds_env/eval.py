import numpy as np
import torch

from dqn_model import Net
from dqn_parallel import MONITOR_HEIGHT, MONITOR_WIDTH
from gym_teeworlds import TeeworldsEnv, NUMBER_OF_IMAGES, OBSERVATION_SPACE, Action
from utils import Monitor, ACTIONS, load_config

config = load_config()
path_to_teeworlds = str(config['path_to_teeworlds'])
set_priority = bool(config.get('set_priority', False))


MODEL_NAME = 'v1'
MODEL_PATH = f'models/{MODEL_NAME}.dat'


def main():
    monitor = Monitor(50, 50, MONITOR_WIDTH, MONITOR_HEIGHT)
    env = TeeworldsEnv(
        monitor=monitor,
        path_to_teeworlds=path_to_teeworlds,
        # server_tick_speed=200,
        episode_duration=20.0,
        map_name='level_2',
    )

    state = env.reset()
    env.set_last_reset()

    net = Net(OBSERVATION_SPACE.shape, n_actions=len(ACTIONS)).to('cpu')
    net.load_state_dict(torch.load(MODEL_PATH))

    while True:
        state_a = np.array(
            state,
            copy=False,
            dtype=np.float32
        ).reshape(
            (1, NUMBER_OF_IMAGES, MONITOR_WIDTH, MONITOR_HEIGHT)
        )
        # noinspection PyUnresolvedReferences,PyCallingNonCallable
        state_v = torch.tensor(state_a, dtype=torch.float32).to('cpu')

        q_values_v = net(state_v)  # calculate q values
        index = torch.argmax(q_values_v)  # get index of value with best outcome

        action = ACTIONS[index]  # extracting action from index ([0,-1], [0,0], [0,1]) <- (0, 1, 2)

        state, reward, is_done, _ = env.step(Action.from_list(action))

        if is_done:
            state = env.reset()


if __name__ == '__main__':
    main()
