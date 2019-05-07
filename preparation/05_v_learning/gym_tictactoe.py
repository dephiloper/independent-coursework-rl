import random
from typing import Tuple

import gym
from gym.spaces import Discrete

WINNING_COMBOS = [7, 56, 73, 84, 146, 273, 292, 448]
FULL_BOARD = 511


class TicTacToeEnvironment(gym.Env):
    def __init__(self, board_width=3, board_height=3):
        self._player_0: int = 0
        self._player_1: int = 0
        self._current_player = 0
        self.action_space = Discrete(9)
        self.observation_space = Discrete(9)
        self.board_width = board_width
        self.board_height = board_height

    metadata = {'render.modes': ['human']}

    def reset(self):
        self._player_0: int = 0
        self._player_1: int = 0
        self._current_player = 0
        return self._player_0, self._player_1

    def render(self, mode='human'):
        for i in range(self.board_width):
            for j in range(self.board_height):
                if 2 ** (i * self.board_width + j) & self._player_0 != 0:
                    print("[X]", end='')
                elif 2 ** (i * self.board_width + j) & self._player_1 != 0:
                    print("[O]", end='')
                else:
                    print("[ ]", end='')
            print()

        print()

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        observation, reward, is_done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            is_done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        valid = self._take_action(action)
        if not valid:
            print("invalid: player", self._current_player)
        # self.status = self.env.step()
        reward = self._get_reward()
        observation = (self._player_0, self._player_1)
        is_done = self._check_if_done(reward)
        self._current_player = (self._current_player + 1) % 2  # alternate between players
        return observation, reward, is_done or not valid, {}

    def _check_if_done(self, reward) -> bool:
        if reward == 1:
            return True
        if FULL_BOARD & (self._player_0 | self._player_1) == FULL_BOARD:
            return True

        return False

    def _take_action(self, action) -> bool:
        assert 0 <= action <= 8
        position = 2 ** action
        if position & (self._player_0 | self._player_1) != position:
            if self._current_player == 0:
                self._player_0 ^= position
            if self._current_player == 1:
                self._player_1 ^= position
            return True
        else:
            return False

    def _get_reward(self) -> int:
        for combo in WINNING_COMBOS:
            if self._current_player == 0 and combo & self._player_0 == combo:
                return 1
            elif self._current_player == 1 and combo & self._player_1 == combo:
                return 1

        return 0
