import collections
import math
import random
import copy
from typing import Tuple, List

from gym_tictactoe import TicTacToeEnvironment

BOARD_WIDTH = 3
BOARD_HEIGHT = 3
env = TicTacToeEnvironment()
saved_action = -1
wanted_depth = 9


def random_move(obs: Tuple):
    return random.choice(possible_moves(obs))


def possible_moves(obs: Tuple) -> List[int]:
    moves = []
    for i in range(BOARD_WIDTH):
        for j in range(BOARD_HEIGHT):
            if 2 ** (i * BOARD_WIDTH + j) & (obs[0][0] | obs[0][1]) == 0:
                moves.append(i * BOARD_WIDTH + j)

    return moves


def min(obs: Tuple, depth: int, alpha: float, beta: float):
    global env, saved_action

    print("min alpha:", alpha, "beta:", beta)

    moves = possible_moves(obs)
    if depth == 0 or len(moves) == 0 or obs[1] != 0:
        return obs[1]
    min_score = beta
    env_copy = copy.copy(env)
    for move in moves:
        obs = env.step(move)
        score = max(obs, depth - 1, alpha, min_score)
        env = copy.copy(env_copy)
        if score < min_score:
            min_score = score
            if min_score <= alpha:
                break

    return min_score


def max(obs: Tuple, depth: int, alpha: float, beta: float):
    global env, saved_action

    print("max alpha:", alpha, "beta:", beta)

    moves = possible_moves(obs)
    if depth == 0 or len(moves) == 0 or obs[1] != 0:
        return -obs[1]
    max_score = alpha
    env_copy = copy.copy(env)
    for move in moves:
        obs = env.step(move)
        score = min(obs, depth - 1, max_score, beta)
        env = copy.copy(env_copy)
        if score > max_score:
            max_score = score

            if max_score >= beta:
                break

            if depth == wanted_depth:
                saved_action = move
                print("saved_action",saved_action)

    return max_score


if __name__ == "__main__":
    observation = (env.reset(), 0, False, None)
    #env.reset()
    #env._player_0 = 3
    #env._player_1 = 36
    #env._current_player = 0
    #observation = ((env._player_0, env._player_1), 0, False, None)
    #env.render()


    while not observation[2]:
        # action = random_move((observation, 0, False))

        _ = max(observation, wanted_depth, -math.inf, math.inf)
        if saved_action != -1:
            observation = env.step(saved_action)
            env.render()
            wanted_depth -= 1

            if not observation[2]:
                observation = env.step(int(input("step:")))
                env.render()
                wanted_depth -= 1
