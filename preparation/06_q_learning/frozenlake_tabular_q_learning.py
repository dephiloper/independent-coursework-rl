import gym
import collections
import random

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
ALPHA = 0.2  # learning rate
TEST_EPISODES = 20
EPSILON_DECAY_LAST_FRAME = 10 ** 3
EPSILON_START = 1.0
EPSILON_FINAL = 0.02


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)  # we only need values, no transition or reward tables

    def sample_env(self, epsilon: float = 0.0):
        action = self.env.action_space.sample() if random.random() < epsilon else self.best_value_and_action(self.state)[1]
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action

        return best_value, best_action

    # also called bellman update
    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v  # bellman approximation
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1 - ALPHA) + new_val * ALPHA  # blending new value over old value

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done:
                break
            state = new_state

        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    frame_idx = 0
    ts_frame = 0
    epsilon = EPSILON_START

    iter_no = 0
    best_reward = 0.0
    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)  # decrease epsilon
        iter_no += 1
        s, a, r, next_s = agent.sample_env(epsilon)
        agent.value_update(s, a, r, next_s)
        ts_frame = frame_idx
        #print(epsilon)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        if reward > 0.8:
            print("Solved in %d iterations!" % iter_no)
            break
