import struct
import subprocess
import time
from collections import deque

import cv2
import screeninfo

import gym
import numpy as np
import zmq
from future.moves import collections
from gym.spaces import Box, Discrete
from mss import mss

NUMBER_OF_IMAGES = 4
STEP_INTERVAL = 1 / 60
EPISODE_DURATION = 5
RESET_DURATION = 1

ARMOR_REWARD = 1
HEALTH_REWARD = 10

start_mon = {'top': 1, 'left': 1, 'width': 84, 'height': 84}
_starting_port = 5000
_open_window_count = 0


class Action:
    def __init__(self):
        self.mouse_x = 0
        self.mouse_y = 0
        self.jump = False
        self.hook = False
        self.shoot = False
        self.direction = 0  # -1, 0, 1
        self.reset = False

    def to_bytes(self) -> bytes:
        action_mask = 0
        action_mask += 1 if self.jump else 0
        action_mask += 2 if self.hook else 0
        action_mask += 4 if self.shoot else 0
        action_mask += 8 if self.direction == 1 else 0
        action_mask += 16 if self.direction == -1 else 0
        action_mask += 32 if self.reset else 0
        return struct.pack("!hhB", self.mouse_x, self.mouse_y, action_mask)

    @staticmethod
    def from_list(list_action: list):
        action = Action()
        action.direction = list_action[0]
        action.jump = bool(list_action[1])
        return action


reset_action = Action()
reset_action.reset = True

with open('config.txt', 'r') as f:
    line = f.readline()
    path_to_teeworlds = line.strip()


class GameInformation:
    def __init__(self, x_position, y_position, armor_collected, health_collected):
        self.x_position = x_position
        self.y_position = y_position
        self.armor_collected = armor_collected
        self.health_collected = health_collected

    def to_dict(self):
        return {
            "x": self.x_position,
            "y": self.y_position,
            "armor_collected": self.armor_collected,
            "health_collected": self.health_collected
        }

    def get_reward(self):
        reward = ARMOR_REWARD * self.armor_collected + HEALTH_REWARD * self.health_collected
        return reward

    def clear(self):
        self.health_collected = 0
        self.armor_collected = 0

    def is_done(self):
        return bool(self.health_collected)


class TeeworldsEnv(gym.Env):
    """
    Creates a single teeworlds environment (one tee, no competitors).

    :arg actions_port specifies the port for sending actions to the teeworlds client
    :arg game_information_port specifies the port for receiving game information like player position
         and if the player has been shot or shot somebody
    :arg teeworlds_srv_port specifies the port the teeworlds server is running on
    :arg ip specifies the ip address of the teeworlds server (use * for running locally)
    """

    def __init__(
            self,
            mon,
            actions_port="5000",
            game_information_port="5001",
            teeworlds_srv_port="8303",
            ip="*",
            map_name="level_0"
    ):
        self.game_information = GameInformation(-1, -1, 0, 0)
        self.mon = mon.copy()
        self.observation_space = Box(0, 255, [NUMBER_OF_IMAGES, self.mon['width'], self.mon['height']])
        self.action_space = Discrete(3)
        self.image_buffer = collections.deque(maxlen=NUMBER_OF_IMAGES)
        self.mon["top"] = self.mon["top"] + 40

        # init video capturing
        self.sct = mss()

        # start server
        subprocess.Popen(
            [
                "./teeworlds_srv",
                "sv_rcon_password 123",
                "sv_max_clients_per_ip 16",
                "sv_map {}".format(map_name),
                "sv_register 1",
                "game_information_port {}".format(game_information_port),
                "sv_port {}".format(teeworlds_srv_port)
            ], cwd=path_to_teeworlds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        time.sleep(2)

        # start client
        subprocess.Popen(
            [
                "{}teeworlds".format(path_to_teeworlds),
                "gfx_screen_width {}".format(self.mon["width"]),
                "gfx_screen_height {}".format(self.mon["height"]),
                "gfx_screen_x {}".format(self.mon["left"]),
                "gfx_screen_y {}".format(self.mon["top"]),
                "snd_volume 0",
                "gfx_fullscreen 0",
                "gfx_borderless 1",
                "cl_skip_start_menu 1",
                "actions_port {}".format(actions_port),
                "connect localhost:{}".format(teeworlds_srv_port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        # create network context
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        address = "tcp://{}:{}".format(ip, actions_port)
        self.socket.bind(address)

        self.game_information_socket = context.socket(zmq.PULL)
        game_information_address = "tcp://{}:{}".format("localhost", game_information_port)
        self.game_information_socket.connect(game_information_address)

        self.last_step_timestamp = time.time()
        self._last_reset = time.time()

        # waiting for everything to build up
        time.sleep(2)

    def _try_fetch_game_state(self):
        """
        Fetches new game information from game_information_socket and updates the x_position and y_position.
        """
        msg = None
        while True:
            try:
                msg = self.game_information_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                break
        if msg:
            armor_collected, health_collected = struct.unpack('<ii', msg)
            # todo check if it is possible for the player to go in an negative area
            self.game_information = GameInformation(0, 0, armor_collected, health_collected)

    def step(self, action: Action, wait_time=0.03):
        """
        Performs a step in the environment by executing the passed action.

        :param action: defines what action to take on the current step
        :param wait_time: TODO
        :return: tuple of observation, reward, done, game_information
        """
        self._wait_for_frame()
        self.socket.send(action.to_bytes())
        # time.sleep(STEP_INTERVAL)
        img = np.asarray(self.sct.grab(self.mon))
        observation = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        if len(self.image_buffer) != NUMBER_OF_IMAGES:
            self.image_buffer = deque([observation] * NUMBER_OF_IMAGES)
        else:
            self.image_buffer.pop()
            self.image_buffer.appendleft(observation)

        self._try_fetch_game_state()

        observation = np.array(self.image_buffer)
        reward = self.game_information.get_reward()
        done = self.game_information.is_done()

        self.game_information.clear()

        if time.time() - self._last_reset > EPISODE_DURATION:
            return observation, 0, True, self.game_information.to_dict()

        return observation, reward, done, self.game_information.to_dict()

    def _wait_for_frame(self):
        t = time.time()
        next_step_timestamp = self.last_step_timestamp + STEP_INTERVAL
        diff = next_step_timestamp - t
        if diff > 0:
            time.sleep(diff)

        self.last_step_timestamp = t

    def reset(self):
        self.image_buffer.clear()
        self.socket.send(reset_action.to_bytes())
        self.game_information.clear()
        time.sleep(RESET_DURATION)
        self._last_reset = time.time()

        return self.step(Action())[0]

    def render(self, mode='human'):
        pass


class TeeworldsMultiEnv(gym.Env):
    def __init__(self, n: int, teeworlds_srv_port="8303"):
        """
        Creates multiple teeworlds environments on the same teeworlds server (n tee's).
        :param n: number of environments / clients / tee's
        :param teeworlds_srv_port: port of the teeworlds server
        """
        global _starting_port
        global _open_window_count

        # setup for window positioning
        top_spacing = 30
        # top_spacing = 0
        mon = start_mon.copy()
        mon["width"] = int(screeninfo.get_monitors()[0].width / 4)
        mon["height"] = int((screeninfo.get_monitors()[0].height - top_spacing) / 4)

        self.envs = []
        self.env_id = 0

        for i in range(_open_window_count, n + _open_window_count):
            # window position adjustments
            mon["left"] = i % 4 * mon["width"]
            mon["top"] = int(i / 4) * mon["height"] + top_spacing

            self.envs.append(
                TeeworldsEnv(
                    mon=mon,
                    actions_port=str(_starting_port),
                    game_information_port=str(_starting_port + 1),
                    teeworlds_srv_port=teeworlds_srv_port
                )
             )
            _open_window_count += 1
            _starting_port += 2

    def step(self, action: Action, wait_time=0.03):
        """
        Performs a step in one of the environments round robin principle.
        :param action: defines what action to take on the current step in the sampled environment
        :param wait_time: TODO!
        :return: tuple of observation, reward, done, game_information from this environment
        """
        index = self.env_id
        self.env_id = (self.env_id + 1) % len(self.envs)
        return self.envs[index].step(action, wait_time)

    def step_by_id(self, action: Action, index: int, wait_time=0.03):
        """
        Performs a step in the selected environment.
        :param action: defines what action to take
        :param index: identifier of the environment in which the action must be executed
        :param wait_time: TODO!
        :return: tuple of observation, reward, done, game_information from this environment
        """
        return self.envs[index].step(action, wait_time)

    def reset(self):
        self.envs[0].reset()

    def render(self, mode='human'):
        pass
