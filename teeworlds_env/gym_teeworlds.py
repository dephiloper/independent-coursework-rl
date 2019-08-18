import os
import struct
import subprocess
import time
from collections import deque

import cv2
import gym
import numpy as np
import zmq
from future.moves import collections
from gym.spaces import Box, Discrete
from mss import mss

from utils import Monitor, mon_iterator, random_id

NUMBER_OF_IMAGES = 4
EPISODE_DURATION = 3
RESET_DURATION = 0.8

ARMOR_REWARD = 1
HEALTH_REWARD = 10

start_mon = {'top': 1, 'left': 1, 'width': 84, 'height': 84}
_starting_port = 5000
_open_window_count = 0

OBSERVATION_SPACE = Box(0, 255, [NUMBER_OF_IMAGES, start_mon['width'], start_mon['height']])
ACTION_SPACE = Discrete(3)


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


def teeworlds_env_iterator(n, monitor_width, monitor_height, top_spacing=0, server_tick_speed=50):
    actions_port = 5000
    teeworlds_server_port = 8303

    for monitor in mon_iterator(n, monitor_width, monitor_height, top_spacing=top_spacing):
        yield TeeworldsEnv(
            monitor,
            str(actions_port),
            str(actions_port+1),
            str(teeworlds_server_port),
            server_tick_speed=server_tick_speed
        )

        actions_port += 2
        teeworlds_server_port += 1


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
            monitor: Monitor,
            actions_port="5000",
            game_information_port="5001",
            teeworlds_srv_port="8303",
            ip="*",
            server_tick_speed=50,
            map_name="level_1",
            device="cpu",
            is_human=False
    ):
        self.game_information = GameInformation(-1, -1, 0, 0)
        self.monitor = monitor
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE
        self.image_buffer = collections.deque(maxlen=NUMBER_OF_IMAGES)
        self.step_interval = 10 / server_tick_speed

        # init video capturing
        self.sct = mss()

        # logging
        os.makedirs("logs/", exist_ok=True)

        log_id = random_id(5)
        server_log = open(f"logs/{log_id}-server_log.txt", "w")
        client_log = open(f"logs/{log_id}-client_log.txt", "w")

        print(f"creating logs at {log_id}")

        sv_properties = [
            "./teeworlds_srv",
            "sv_rcon_password 123",
            "sv_max_clients_per_ip 16",
            f"sv_map {map_name}",
            "sv_register 1",
            f"game_information_port {game_information_port}",
            f"sv_port {teeworlds_srv_port}",
            f"tick_speed {server_tick_speed}",
        ]

        # start server
        subprocess.Popen(
            sv_properties,
            cwd=path_to_teeworlds,
            stdout=server_log,
            stderr=subprocess.STDOUT
        )

        time.sleep(1)

        c_properties = [
            f"./teeworlds",
            f"gfx_screen_width {self.monitor.width}",
            f"gfx_screen_height {self.monitor.height}",
            f"gfx_screen_x {self.monitor.left}",
            f"gfx_screen_y {self.monitor.top}",
            "snd_volume 0",
            "gfx_fullscreen 0",
            "gfx_borderless 1",
            "cl_skip_start_menu 1",
            f"actions_port {actions_port}",
            f"connect localhost:{teeworlds_srv_port}",
            f"tick_speed {server_tick_speed}"
        ]

        if device == "cuda":
            c_properties.insert(0, "optirun")

        if is_human:
            c_properties.append("--human")

        # start client
        subprocess.Popen(
            c_properties,
            cwd=path_to_teeworlds,
            stdout=client_log,
            stderr=subprocess.STDOUT
        )

        # create network context
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        address = f"tcp://{ip}:{actions_port}"
        self.socket.bind(address)

        self.game_information_socket = context.socket(zmq.PULL)
        game_information_address = f"tcp://localhost:{game_information_port}"
        self.game_information_socket.connect(game_information_address)

        self.last_step_timestamp = time.time()
        self._last_reset = time.time()

        # waiting for everything to build up
        time.sleep(1)

    def _try_fetch_game_state(self):
        """
        Fetches new game information from game_information_socket and updates the left and top.
        """
        while True:
            try:
                msg = self.game_information_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                break
            if msg:
                armor_collected, health_collected = struct.unpack('<ii', msg)

                # todo check if it is possible for the player to go in an negative area
                self.game_information.armor_collected += armor_collected
                self.game_information.health_collected += health_collected

    def step(self, action: Action):
        """
        Performs a step in the environment by executing the passed action.

        :param action: defines what action to take on the current step
        :return: tuple of observation, reward, done, game_information
        """
        self._wait_for_frame()

        self.socket.send(action.to_bytes())

        img = np.asarray(self.sct.grab(self.monitor.to_dict()))
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
        info = self.game_information.to_dict()
        self.game_information.clear()

        if observation is None:
            print('None observation')

        if time.time() - self._last_reset > EPISODE_DURATION:
            return observation, 0, True, info

        return observation, reward, done, info

    def _wait_for_frame(self):
        current = time.time()
        next_step_timestamp = self.last_step_timestamp + self.step_interval
        diff = next_step_timestamp - current
        if diff > 0:
            time.sleep(diff)

        self.last_step_timestamp = max(next_step_timestamp, current)

    def reset(self, reset_duration=RESET_DURATION):
        self.image_buffer.clear()
        self.socket.send(reset_action.to_bytes())
        self.game_information.clear()
        time.sleep(reset_duration)
        self._last_reset = time.time()

        observation, reward, done, info = self.step(Action())
        if observation is None:
            print('None state')

        return observation

    def render(self, mode='human'):
        pass