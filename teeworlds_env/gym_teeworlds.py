import struct
import subprocess
import time
import screeninfo

import gym
import numpy
import zmq
from gym.spaces import Box
from mss import mss

mon = {'top': 1, 'left': 1, 'width': 480, 'height': 320}
info = {'x': -1, 'y': -1, 'got_hit': False, 'enemy_hit': False}
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

    def to_bytes(self) -> bytes:
        action_mask = 0
        action_mask += 1 if self.jump else 0
        action_mask += 2 if self.hook else 0
        action_mask += 4 if self.shoot else 0
        action_mask += 8 if self.direction == 1 else 0
        action_mask += 16 if self.direction == -1 else 0

        return struct.pack("!hhB", self.mouse_x, self.mouse_y, action_mask)


with open('config.txt', 'r') as f:
    l = f.readline()
    path_to_teeworlds = l.strip()


class TeeworldsEnv(gym.Env):
    """
    Creates a single teeworlds environment (one tee, no competitors).

    :arg actions_port specifies the port for sending actions to the teeworlds client
    :arg game_information_port specifies the port for receiving game information like player position
         and if the player has been shot or shot somebody
    :arg teeworlds_srv_port specifies the port the teeworlds server is running on
    :arg ip specifies the ip address of the teeworlds server (use * for running locally)
    """

    def __init__(self, actions_port="5000", game_information_port="5001", teeworlds_srv_port="8303", ip="*"):
        self.observation_space = Box(0, 255, [mon['width'], mon['height'], 3])
        self.action_space = Box(-1, 1, [5, ])

        # init video capturing
        self.sct = mss()

        # start server
        subprocess.Popen(
            [
                "{}teeworlds_srv".format(path_to_teeworlds),
                "sv_max_clients_per_ip 16",
                "sv_port {}".format(teeworlds_srv_port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        time.sleep(0.2)

        # start client
        subprocess.Popen(
            [
                "{}teeworlds".format(path_to_teeworlds),
                "gfx_screen_width {}".format(mon["width"]),
                "gfx_screen_height {}".format(mon["height"]),
                "gfx_screen_x {}".format(mon["left"]),
                "gfx_screen_y {}".format(mon["top"]),
                "snd_volume 0",
                "gfx_fullscreen 0",
                "gfx_borderless 1",
                "cl_skip_start_menu 1",
                "actions_port {}".format(actions_port),
                "game_information_port {}".format(game_information_port),
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

        # waiting for everything to build up
        time.sleep(2)

    def _try_update_position(self):
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
            x, y = struct.unpack('<ii', msg)
            # todo check if it is possible for the player to go in an negative area
            if x != -1 and y != -1:
                info["x"] = x
                info["y"] = y

    def step(self, action: Action):
        """
        Performs a step in the environment by executing the passed action.
        :param action: defines what action to take on the current step
        :return: tuple of observation, reward, done, info
        """
        self.socket.send(action.to_bytes())
        observation = numpy.asarray(self.sct.grab(mon))

        self._try_update_position()

        reward = 0
        done = 0
        return observation, reward, done, info

    def reset(self):
        pass

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
        mon["width"] = int(screeninfo.get_monitors()[0].width / 4)
        mon["height"] = int((screeninfo.get_monitors()[0].height - top_spacing) / 4)

        self.envs = []
        self.env_id = 0

        for i in range(_open_window_count, n + _open_window_count):
            # window position adjustments
            mon["left"] = i % 4 * mon["width"]
            mon["top"] = int(i / 4) * mon["height"] + top_spacing

            print(_starting_port)

            self.envs.append(TeeworldsEnv(actions_port=str(_starting_port),
                                          game_information_port=str(_starting_port + 1),
                                          teeworlds_srv_port=teeworlds_srv_port))
            _open_window_count += 1
            _starting_port += 2

    def step(self, action: Action):
        """
        Performs a step in one of the environments round robin principle.
        :param action: defines what action to take on the current step in the sampled environment
        :return: tuple of observation, reward, done, info from this environment
        """
        index = self.env_id
        self.env_id = (self.env_id + 1) % len(self.envs)
        return self.envs[index].step(action)

    def step_by_id(self, action: Action, index: int):
        """
        Performs a step in the selected environment.
        :param action: defines what action to take
        :param index: identifier of the environment in which the action must be executed
        :return: tuple of observation, reward, done, info from this environment
        """
        return self.envs[index].step(action)

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
