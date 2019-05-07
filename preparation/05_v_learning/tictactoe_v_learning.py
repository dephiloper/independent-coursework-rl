import collections
import random
from typing import Tuple

from gym_tictactoe import TicTacToeEnvironment

BOARD_WIDTH = 3
BOARD_HEIGHT = 3
GAMMA = 0.9
TEST_EPISODES = 100


class Agent:
    def __init__(self):
        self.env = TicTacToeEnvironment()
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)  # don't know if needed
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = random_move(self.state)
            new_state, rew, is_done, _ = self.env.step(action)

            # enemy step
            if not is_done:
                new_state, _, is_done, _ = self.env.step(random_move(new_state))

            self.rewards[(self.state, action, new_state)] = rew
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for target_state, count in target_counts.items():
            rew = self.rewards[(state, action, target_state)]
            action_value += (count / total) * (rew + GAMMA * self.values[target_state])  # bellman eq

        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, rew, is_done, _ = env.step(action)

            # enemy step
            if not is_done:
                new_state, _, is_done, _ = self.env.step(random_move(new_state))

            self.rewards[(state, action, new_state)] = rew
            self.transits[(state, action)][new_state] += 1
            total_reward += rew
            if is_done:
                break
            state = new_state

        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            state_values = [self.calc_action_value(state, action)
                            for action in range(self.env.action_space.n)]
            self.values[state] = max(state_values)


def random_move(observation: Tuple):
    moves = []
    for i in range(BOARD_WIDTH):
        for j in range(BOARD_HEIGHT):
            if 2 ** (i * BOARD_WIDTH + j) & (observation[0] | observation[1]) == 0:
                moves.append(i * BOARD_WIDTH + j)

    return random.choice(moves)


if __name__ == "__main__":
    test_env = TicTacToeEnvironment()
    agent = Agent()
    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        if reward > best_reward:
            print("best reward updated:", best_reward, "->", reward)
            best_reward = reward
            if reward > 0.8:
                print("solved in %d iterations!" % iter_no)
                break
