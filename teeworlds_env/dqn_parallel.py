import os
import time
from queue import Empty
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.optim
from tensorboardX import SummaryWriter
from torch.multiprocessing import Process, Value, Queue, Event
from torch.nn import Linear
from tqdm import tqdm

from dqn_model import Net, NoisyLinear, DuelingNet
from gym_teeworlds import teeworlds_env_settings_iterator, OBSERVATION_SPACE, TeeworldsEnvSettings, Action, \
    NUMBER_OF_IMAGES
from utils import ExperienceBuffer, ACTIONS, ACTION_LABELS, Experience, load_config, PriorityExperienceBuffer, \
    ExploringStrategy


# exp collecting
NUM_WORKERS = 4
COLLECT_EXPERIENCE_SIZE = 2000  # init: 2000 (amount of experiences to collect after each training step)
GAME_TICK_SPEED = 100  # default: 50 (game speed, when higher more screenshots needs to be captures)
EPISODE_DURATION = 40  # default: 40
MONITOR_WIDTH = 84  # init: 84 width of game screen
MONITOR_HEIGHT = 84  # init: 84 height of game screen (important for conv)
MONITOR_X_PADDING = 20
MONITOR_Y_PADDING = 20
PRIORITY_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 10 ** 5
PRINT_EXPERIENCE_BUFFER = False


DOUBLE_DQN = True
EXPERIENCE_BUFFER_CLASS = PriorityExperienceBuffer

EXPLORING_STRATEGY = ExploringStrategy.NOISY_NETWORK
LINEAR_LAYER_CLASS = Linear if EXPLORING_STRATEGY == ExploringStrategy.EPSILON_GREEDY else NoisyLinear
NET_TYPE = DuelingNet  # use DuelingNet for DuelingDQN and Net for default

MIN_EPSILON = 0.02  # init: 0.02
EPSILON_START = 1.0  # init: 1.0
EPSILON_DECAY = 0.01  # init: 0.01

# training
REPLAY_START_SIZE = 4000  # init: 10000 (min amount of experiences in replay buffer before training starts)
REPLAY_SIZE = 10000  # init: 10000 (max capacity of replay buffer)
DEVICE = 'cuda'  # init: 'cpu'
BATCH_SIZE = 512  # init: 32 (sample size of experiences from replay buffer)
NUM_TRAININGS_PER_EPOCH = 50  # init: 50 (amount of BATCH_SIZE x NUM_TRAININGS_PER_EPOCH will be trained)
GAMMA = 0.99  # init: .99 (bellman equation)
LEARNING_RATE = 1e-4  # init: 1e-4 (also quite low eventually using default 1e-3)
SYNC_TARGET_FRAMES = COLLECT_EXPERIENCE_SIZE * 5  # init: 10000 (how frequently we sync target net with net)
L2_REGULARIZATION = 1e-5  # init: 1e-5
MAP_NAMES = ['newlevel_0', 'newlevel_1', 'newlevel_2', 'newlevel_3']

# evaluation
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100
MODEL_SAVE_INTERVAL = 100000

MEAN_REWARD_BOUND = 12

config = load_config()
path_to_teeworlds = str(config['path_to_teeworlds'])
set_priority = bool(config.get('set_priority', False))
session_name = str(config['session_name'])

MODEL_NAME = f'model_{session_name}'
TENSORBOARD_LOG_DIR = f'./runs/{session_name}'

if os.path.isdir(TENSORBOARD_LOG_DIR):
    raise AssertionError(f'log dir for tensorboard already exists \"{TENSORBOARD_LOG_DIR}\"')


class GameStats:
    def __init__(self, reward):
        self.reward = reward


class Worker(Process):
    def __init__(
            self,
            worker_index,
            env_settings: TeeworldsEnvSettings,
            experience_queue: Queue,
            stats_queue: Queue,
            net: Union[Net, DuelingNet],
            epsilon: Value,
            action_list: List,
            device: str = 'cpu'
    ):
        Process.__init__(self)

        self.worker_index = worker_index
        self.experience_queue = experience_queue
        self.stats_queue = stats_queue
        self.net = net
        self.epsilon = epsilon
        self.actions = action_list
        self.device = device

        self.env_settings = env_settings
        self.total_reward = 0

        self._stop_on_done = Value('b', False)
        self.stopped = Event()  # type: Event

        self._running_queue = Queue()
        self._running_queue.put(False)  # do not start immediately
        self.env = None
        self.state = None

    def start_collecting_experience(self):
        self._running_queue.put(True)

    def stop_collecting_experience(self):
        self._stop_on_done.value = True

    def initialize_env(self):
        self.env = self.env_settings.create_env()

    def _idle_for_running(self):
        try:
            token = self._running_queue.get_nowait()

            # if False was in queue
            if not token:
                self.stopped.set()
                self.state = self.env.reset()
                # wait for next true
                while True:
                    token = self._running_queue.get()
                    if token:
                        break
                    self.state = self.env.reset()
                self.env.set_last_reset()
        except Empty:
            pass

        self.stopped.clear()

    def _do_step(self):
        self._idle_for_running()

        if EXPLORING_STRATEGY == ExploringStrategy.EPSILON_GREEDY and np.random.random() < self.epsilon.value:
            index = np.random.randint(len(self.actions))
        else:
            state_a = np.array(self.state, copy=False, dtype=np.float32).reshape(
                (1, NUMBER_OF_IMAGES, self.env.monitor.width, self.env.monitor.height))

            # noinspection PyUnresolvedReferences,PyCallingNonCallable
            state_v = torch.tensor(state_a, dtype=torch.float32).to(self.device)

            q_values_v = self.net(state_v)  # calculate q values
            index = int(torch.argmax(q_values_v))  # get index of value with best outcome

        action = self.actions[index]  # extracting action from index ([0,-1], [0,0], [0,1]) <- (0, 1, 2)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(Action.from_list(action))

        assert (new_state is not None)

        self.total_reward += reward

        # store experience in exp_buffer
        exp = Experience(self.state, index, reward, is_done, new_state, self.worker_index)
        self.experience_queue.put(exp)

        self.state = new_state

        # end of episode situation
        if is_done:
            self.state = self.env.reset()
            self.stats_queue.put(GameStats(self.total_reward))
            self.total_reward = 0
            if self._stop_on_done.value:
                self._running_queue.put(False)
                self._stop_on_done.value = False

    # noinspection PyCallingNonCallable,PyUnresolvedReferences
    def run(self) -> None:
        self.initialize_env()

        while True:
            try:
                self._do_step()
            except Exception as e:
                print(str(e))
                break


def setup():
    assert (SYNC_TARGET_FRAMES % COLLECT_EXPERIENCE_SIZE == 0)
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs('saves', exist_ok=True)

    # wait for process setting
    while set_priority and os.nice(0) == 0:
        time.sleep(0.2)


def print_experience_buffer(experience_buffer: Union[ExperienceBuffer, PriorityExperienceBuffer]):
    current_index = None
    for index, experience in enumerate(experience_buffer.buffer):
        assert (type(experience) == Experience)
        if experience.reward:
            current_index = index

    if current_index is None:
        return

    worker_index = experience_buffer.buffer[current_index].worker_index

    while True:
        experience = experience_buffer.buffer[current_index]

        img = np.concatenate(experience.new_state, axis=1)

        cv2.imshow(f'{current_index}: {ACTION_LABELS[experience.action]} reward={experience.reward}', img)
        key = cv2.waitKey()
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            break
        elif key == 97:  # A
            new_index = current_index
            while new_index > 0:
                new_index -= 1
                if experience_buffer.buffer[new_index].worker_index == worker_index:
                    current_index = new_index
                    break
        elif key == 100:  # D
            new_index = current_index
            while new_index < len(experience_buffer) - 1:
                new_index += 1
                if experience_buffer.buffer[new_index].worker_index == worker_index:
                    current_index = new_index
                    break
        elif key == 119:  # W
            worker_index = (worker_index + 1) % NUM_WORKERS
            new_index = current_index
            while new_index > 0:
                new_index -= 1
                if experience_buffer.buffer[new_index].worker_index == worker_index:
                    current_index = new_index
                    break
        elif key == 115:  # S
            worker_index -= worker_index
            if worker_index < 0:
                worker_index = NUM_WORKERS - 1
            new_index = current_index
            while new_index > 0:
                new_index -= 1
                if experience_buffer.buffer[new_index].worker_index == worker_index:
                    current_index = new_index
                    break

        cv2.destroyAllWindows()


def should_save_model(frame_idx, mean_reward, max_mean_reward, last_frame_model_saved):
    save_because_new_reward = mean_reward > max_mean_reward
    save_because_interval = (mean_reward + 1 > max_mean_reward) and \
                            (last_frame_model_saved + MODEL_SAVE_INTERVAL < frame_idx)
    return save_because_new_reward or save_because_interval


def main():
    setup()

    workers = []
    experience_queue = Queue()
    stats_queue = Queue()

    observation_size = OBSERVATION_SPACE.shape
    epsilon = Value('d', EPSILON_START)

    # noinspection PyUnresolvedReferences
    net = NET_TYPE(observation_size, n_actions=len(ACTIONS), linear_layer_class=LINEAR_LAYER_CLASS).to(DEVICE)
    net.share_memory()

    # noinspection PyUnresolvedReferences
    target_net = NET_TYPE(observation_size, n_actions=len(ACTIONS), linear_layer_class=LINEAR_LAYER_CLASS).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)

    for worker_index, env_setting in enumerate(teeworlds_env_settings_iterator(
            n=NUM_WORKERS,
            path_to_teeworlds=path_to_teeworlds,
            monitor_width=MONITOR_WIDTH,
            monitor_height=MONITOR_HEIGHT,
            top_spacing=40,
            server_tick_speed=GAME_TICK_SPEED,
            episode_duration=EPISODE_DURATION * (50 / GAME_TICK_SPEED),
            monitor_x_padding=MONITOR_X_PADDING,
            monitor_y_padding=MONITOR_Y_PADDING,
            map_names=MAP_NAMES
    )):
        worker = Worker(worker_index, env_setting, experience_queue, stats_queue, net, epsilon, ACTIONS, DEVICE)
        workers.append(worker)

    experience_buffer = EXPERIENCE_BUFFER_CLASS(capacity=REPLAY_SIZE)

    for worker in workers:
        worker.start()
        time.sleep(2)

    time.sleep(4)

    # start workers
    for worker in workers:
        worker.start_collecting_experience()

    frame_idx = 0
    last_frame_model_saved = 0
    beta = BETA_START
    eval_states = None

    writer = SummaryWriter(TENSORBOARD_LOG_DIR)
    game_stats = []

    max_mean_reward = 0
    finished_episodes = 0
    epoch = 0

    while True:
        # collect experience
        for _ in tqdm(range(COLLECT_EXPERIENCE_SIZE), desc='collecting data: '):
            exp = experience_queue.get()
            experience_buffer.append(exp)
            frame_idx += 1
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            # gather stats for logging
            while True:
                try:
                    game_stat = stats_queue.get_nowait()
                    finished_episodes += 1
                except Empty:
                    break

                game_stats.append(game_stat)
                writer.add_scalar('reward', game_stat.reward, frame_idx)

                reward_10 = 0
                for stat in game_stats[-10:]:
                    reward_10 += stat.reward
                reward_10 /= len(game_stats[-10:])

                writer.add_scalar('reward_10', reward_10, frame_idx)

                # optional output
                if EXPLORING_STRATEGY == ExploringStrategy.EPSILON_GREEDY:
                    writer.add_scalar('epsilon', epsilon.value, frame_idx)

                if EXPERIENCE_BUFFER_CLASS == PriorityExperienceBuffer:
                    writer.add_scalar('beta', beta, frame_idx)

        if PRINT_EXPERIENCE_BUFFER:
            for worker in workers:
                worker.stop_collecting_experience()
            print_experience_buffer(experience_buffer)

        # check if buffer is large enough for training else back to collecting training data
        if len(experience_buffer) < REPLAY_START_SIZE:
            continue

        # stop experience collection on all workers
        for worker in workers:
            worker.stop_collecting_experience()

        for worker in workers:
            worker.stopped.wait()

        mean_reward = np.mean([stat.reward for stat in game_stats[-finished_episodes:]])
        if should_save_model(frame_idx, mean_reward, max_mean_reward, last_frame_model_saved):
            max_mean_reward = mean_reward
            last_frame_model_saved = frame_idx
            save_name = f"saves/{MODEL_NAME}_epoch-{epoch:04d}_rew-{max_mean_reward:08.1f}.dat"
            torch.save(net.state_dict(), save_name)
            print(f'saved model "{save_name}" with mean reward {mean_reward:.1f}')
        finished_episodes = 0

        if eval_states is None:
            eval_states = experience_buffer.sample(STATES_TO_EVALUATE)[0][0]
            np.array(eval_states, copy=False)

        # sync nets (copy weights)
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            target_net.load_state_dict(net.state_dict())

        epsilon.value = max(MIN_EPSILON, epsilon.value - EPSILON_DECAY)

        # training
        total_loss = []
        for _ in tqdm(range(NUM_TRAININGS_PER_EPOCH), desc='training: '):
            optimizer.zero_grad()

            batch, batch_weights, batch_indices = experience_buffer.sample(BATCH_SIZE, beta=beta)

            # perform optimization by minimizing the loss
            loss_v, sample_priorities_v = calc_loss(batch, batch_weights, net, target_net,
                                                    device=DEVICE, is_double=DOUBLE_DQN)  # double dqn enabled!
            total_loss.append(loss_v.item())
            loss_v.backward()
            optimizer.step()
            writer.add_scalar('mean_loss', np.mean(total_loss), frame_idx)
            mean_val = calc_values_of_states(eval_states, net, device=DEVICE)
            writer.add_scalar('values_mean', mean_val, frame_idx)
            total_loss.clear()

            if EXPERIENCE_BUFFER_CLASS == PriorityExperienceBuffer:
                experience_buffer.update_priorities(batch_indices, sample_priorities_v.data.cpu().numpy())

                mean, std = net.get_stats()
                writer.add_scalar('net_weight_mean', mean, frame_idx)
                writer.add_scalar('net_weight_std', std, frame_idx)

        if EXPLORING_STRATEGY == ExploringStrategy.NOISY_NETWORK:
            snr_values = net.noisy_layers_sigma_snr()
            for layer_idx, sigma_l2 in enumerate(snr_values):
                writer.add_scalar(f"sigma_snr_layer_{layer_idx + 1}", sigma_l2, frame_idx)

        epoch += 1

        # restart experience collection on all workers
        for worker in workers:
            worker.start_collecting_experience()


# performs the loss calculation mentioned in 6 and 7
# equations:
#   https://bit.ly/2I7iBqa <- steps which aren't end of episode
#   https://bit.ly/2HFsOec <- final steps
def calc_loss(batch, batch_weights, net, tgt_net, device="cpu", is_double=True):
    states, actions, rewards, dones, next_states = batch  # unpack the sample

    # wrap numpy data in torch tensors <- execution on gpu fast like sonic
    states_v = torch.tensor(states, dtype=torch.float32).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
    actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
    # noinspection PyArgumentList
    done_mask = torch.BoolTensor(dones).to(device)

    if batch_weights is not None:
        batch_weights_v = torch.tensor(batch_weights, dtype=torch.float32).to(device)
    else:
        batch_weights_v = None

    # extract q-values for taken actions using the gather function
    # gather explained: https://stackoverflow.com/a/54706716/10547035 or page 144
    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    '''
    # apply target network to next state observations, and calculate the maximum q-value along the same action dim 1
    # max returns both max value and indices of those values
    next_state_values = tgt_net(next_states_v).max(1)[0]
    '''
    # why double? basic DQN has tendency to overestimate values of Q (harm training process)
    # if enabled we calculate the best action to take in the next state using our main trained network
    # but, values corresponding to this action come from the target network
    if is_double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(-1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]

    # this part is mentioned as very important!
    # if transition in the batch is from the last step in the episode, then our value of the action does not
    # have a discounted reward of the next state, as there is no next state to gather reward from
    # without this training will not converge!
    next_state_values[done_mask] = 0.0

    # detach() returns tensor w/o connection to its calculation history
    # detach value from computation graph, which prevents gradients of flowing into target network
    # when not performed back propagation of the loss will affect both nets
    next_state_values = next_state_values.detach()

    # calculate the bellman approximation value
    expected_state_action_values = next_state_values * GAMMA + rewards_v

    # pytorch MSE doesn't support weights -> calculate MSE and multiply batch weights
    if batch_weights_v is not None:
        losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    else:
        losses_v = (state_action_values - expected_state_action_values) ** 2

    return losses_v.mean(), losses_v + 1e-5  # small loss added to handle zero loss situations


# just for comparision of training process w/ and w/o double dqn
# noinspection PyUnresolvedReferences
def calc_values_of_states(states, net, device="cpu"):
    mean_values = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_values.append(best_action_values_v.mean().item())

    return np.mean(mean_values)


if __name__ == '__main__':
    main()
