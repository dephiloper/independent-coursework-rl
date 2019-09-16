import itertools
import os
import struct
import subprocess
import time
from collections import deque
from typing import List

import cv2
import gym
import numpy as np
import zmq
from future.moves import collections
from gym.spaces import Box, Discrete
from mss import mss

from utils import Monitor, mon_iterator, random_id

RESET_DURATION = 0.8
NUMBER_OF_IMAGES = 4

ARMOR_REWARD = 1
HEALTH_REWARD = 5
DIE_REWARD = 0 
GAME_INFORMATION_DELAY = 0

start_mon = {'top': 1, 'left': 1, 'width': 84, 'height': 84}
_starting_port = 5000
_open_window_count = 0
OBSERVATION_SPACE = Box(0, 255, [NUMBER_OF_IMAGES, start_mon['width'], start_mon['height']], dtype=np.uint8)
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


class GameInformation:
    def __init__(self, x_position, y_position, armor_collected, health_collected, died):
        self.x_position = x_position
        self.y_position = y_position
        self.armor_collected = armor_collected
        self.health_collected = health_collected
        self.died = died

    def to_dict(self):
        return {
            "x": self.x_position,
            "y": self.y_position,
            "armor_collected": self.armor_collected,
            "health_collected": self.health_collected,
            "died": self.died
        }

    def get_reward(self):
        if self.died:
            reward = DIE_REWARD
        else:
            reward = ARMOR_REWARD * self.armor_collected + HEALTH_REWARD * self.health_collected
        return reward

    def clear(self):
        self.health_collected = 0
        self.armor_collected = 0
        self.died = False

    def is_done(self):
        return bool(self.health_collected) or self.died

    @staticmethod
    def empty():
        return GameInformation(-1, -1, 0, 0, False)


def teeworlds_env_settings_iterator(
        n,
        path_to_teeworlds: str,
        monitor_width: int,
        monitor_height: int,
        top_spacing: int = 0,
        server_tick_speed: int = 50,
        episode_duration: float = 20.0,
        monitor_x_padding: int = 0,
        monitor_y_padding: int = 0,
        map_names: List[str] = None
):
    actions_port = 5000
    teeworlds_server_port = 8303

    if map_names is None:
        map_names = ['level_0']

    maps = itertools.cycle(map_names)

    for monitor in mon_iterator(
            n,
            monitor_width,
            monitor_height,
            top_spacing=top_spacing,
            x_padding=monitor_x_padding,
            y_padding=monitor_y_padding
    ):
        yield TeeworldsEnvSettings(
            monitor=monitor,
            path_to_teeworlds=path_to_teeworlds,
            actions_port=str(actions_port),
            game_information_port=str(actions_port+1),
            teeworlds_srv_port=str(teeworlds_server_port),
            server_tick_speed=server_tick_speed,
            episode_duration=episode_duration,
            map_name=next(maps)
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
            path_to_teeworlds,
            actions_port="5000",
            game_information_port="5001",
            teeworlds_srv_port="8303",
            ip="*",
            server_tick_speed=50,
            episode_duration: float = 5.0,
            map_name="level_0",
            device="cpu",
            is_human=False,
    ):
        self.monitor = monitor
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE
        self.image_buffer = collections.deque(maxlen=NUMBER_OF_IMAGES)
        self.step_interval = 10 / server_tick_speed
        self.episode_duration = episode_duration
        # init video capturing
        self.sct = mss()

        # logging
        os.makedirs("logs/", exist_ok=True)

        log_id = random_id(5)
        server_log = open(f"logs/{log_id}-server_log.txt", "w")
        client_log = open(f"logs/{log_id}-client_log.txt", "w")

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
        # noinspection PyUnresolvedReferences
        self.socket = context.socket(zmq.PUSH)
        address = f"tcp://{ip}:{actions_port}"
        self.socket.bind(address)

        # noinspection PyUnresolvedReferences
        self.game_information_socket = context.socket(zmq.PULL)
        game_information_address = f"tcp://localhost:{game_information_port}"
        self.game_information_socket.connect(game_information_address)

        self.last_step_timestamp = None
        self._last_reset = time.time()

        self.game_information_deque = None
        self.game_information = None

        # waiting for everything to build up
        time.sleep(1)

    def _try_fetch_game_state(self):
        """
        Fetches new game information from game_information_socket and updates the left and top.
        """
        game_information = GameInformation.empty()
        while True:
            try:
                # noinspection PyUnresolvedReferences
                msg = self.game_information_socket.recv(zmq.DONTWAIT)
            except zmq.Again:
                break
            if msg:
                armor_collected, health_collected, died = struct.unpack('<iiB', msg)

                died = bool(died)

                game_information.armor_collected += armor_collected
                game_information.health_collected += health_collected
                game_information.died = game_information.died or died

        if GAME_INFORMATION_DELAY:
            self.game_information_deque.appendleft(game_information)
        else:
            self.game_information = game_information

    def capture_game_image(self):
        img = np.asarray(self.sct.grab(self.monitor.to_dict()))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    def get_observation(self):
        return np.array(self.image_buffer)

    def step(self, action: Action):
        """
        Performs a step in the environment by executing the passed action.

        :param action: defines what action to take on the current step
        :return: tuple of observation, reward, done, game_information
        """
        self.socket.send(action.to_bytes())

        self._wait_for_frame()

        self._try_fetch_game_state()
        game_image = self.capture_game_image()

        assert len(self.image_buffer) == NUMBER_OF_IMAGES, f"wrong image buffer length: {len(self.image_buffer)}, " \
                                                           f"should be {NUMBER_OF_IMAGES}, check if you reset the env "\
                                                           f"before accessing it "
        self.image_buffer.append(game_image)

        observation = self.get_observation()

        if GAME_INFORMATION_DELAY:
            game_information = self.game_information_deque.pop()
        else:
            game_information = self.game_information

        reward = game_information.get_reward()
        done = game_information.is_done()
        info = game_information.to_dict()

        if time.time() - self._last_reset > self.episode_duration:
            return observation, 0, True, info

        return observation, reward, done, info

    def _wait_for_frame(self):
        current = time.time()
        if self.last_step_timestamp is None:
            self.last_step_timestamp = current
        else:
            next_step_timestamp = self.last_step_timestamp + self.step_interval
            diff = next_step_timestamp - current
            if diff > 0:
                time.sleep(diff)
            else:
                print(f'Frame drop detected: missed {int(diff*-1000)} milliseconds')

            self.last_step_timestamp = max(next_step_timestamp, current)

    def set_last_reset(self):
        self._last_reset = time.time()

    def reset(self, reset_duration=RESET_DURATION):
        self.image_buffer.clear()
        self.socket.send(reset_action.to_bytes())
        if GAME_INFORMATION_DELAY:
            self.game_information_deque = deque([GameInformation.empty()] * GAME_INFORMATION_DELAY)
        else:
            self.game_information = GameInformation.empty()
        time.sleep(reset_duration)
        self.set_last_reset()

        self.image_buffer = deque([self.capture_game_image()] * NUMBER_OF_IMAGES, maxlen=NUMBER_OF_IMAGES)

        self.last_step_timestamp = None

        return self.get_observation()

    def render(self, mode='human'):
        pass


class TeeworldsEnvSettings:
    def __init__(
            self,
            monitor: Monitor,
            path_to_teeworlds: str,
            actions_port="5000",
            game_information_port="5001",
            teeworlds_srv_port="8303",
            ip="*",
            server_tick_speed=50,
            episode_duration: float = 20.0,
            map_name="level_0",
            is_human=False
    ):
        self.monitor = monitor
        self.path_to_teeworlds = path_to_teeworlds
        self.actions_port = actions_port
        self.game_information_port = game_information_port
        self.teeworlds_srv_port = teeworlds_srv_port
        self.ip = ip
        self.server_tick_speed = server_tick_speed
        self.episode_duration = episode_duration
        self.map_name = map_name
        self.is_human = is_human

    def create_env(self):
        return TeeworldsEnv(
            monitor=self.monitor,
            path_to_teeworlds=self.path_to_teeworlds,
            actions_port=self.actions_port,
            game_information_port=self.game_information_port,
            teeworlds_srv_port=self.teeworlds_srv_port,
            ip=self.ip,
            server_tick_speed=self.server_tick_speed,
            episode_duration=self.episode_duration,
            map_name=self.map_name,
            is_human=self.is_human,
        )

    def __str__(self):
        return f'TeeworldsEnvSettings: [' \
               f'\n\tmonitor: {self.monitor}' \
               f'\n\tactions_port={self.actions_port}' \
               f'\n\tgame_information_port={self.game_information_port}'\
               f'\n\tteeworlds_srv_port={self.teeworlds_srv_port}'\
               f'\n\tip={self.ip}'\
               f'\n\tserver_tick_speed={self.server_tick_speed}'\
               f'\n\tmap_name={self.map_name}'\
               f'\n\tis_human={self.is_human}'\
               f'\n]'
