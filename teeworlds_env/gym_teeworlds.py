import struct
import subprocess
import time

import gym
import numpy
import zmq
from gym.spaces import Box
from mss import mss

mon = {'top': 1, 'left': 1, 'width': 480, 'height': 320}


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
    def __init__(self, action_port: str, game_information_port: str, ip="*"):
        self.observation_space = Box(0, 255, [mon['width'], mon['height'], 3])
        self.action_space = Box(-1, 1, [5, ])

        self.x_position = -1
        self.y_position = -1

        # init video capturing
        self.sct = mss()

        # start server
        subprocess.Popen([path_to_teeworlds + "teeworlds_srv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(0.2)

        # start client
        subprocess.Popen(
            [
                "{}teeworlds".format(path_to_teeworlds),
                "gfx_screen_width {}".format(mon["width"]),
                "gfx_screen_height {}".format(mon["height"]),
                "gfx_screen_x {}".format(mon["left"]),
                "gfx_screen_y {}".format(mon["top"]),
                "gfx_fullscreen 0",
                "gfx_borderless 1",
                "cl_skip_start_menu 1",
                "connect localhost:8303"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )

        # create network context
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        address = "tcp://{}:{}".format(ip, action_port)
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
            self.x_position = x
            self.y_position = y

    def step(self, action: Action):
        self.socket.send(action.to_bytes())
        observation = numpy.asarray(self.sct.grab(mon))

        self._try_update_position()

        # reward, done, info = self.socket.recv()
        return observation  # , reward, done, info

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
