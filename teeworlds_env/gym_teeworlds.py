import struct
import subprocess
import time

import gym
import numpy
import zmq
from gym.spaces import Box
from mss import mss

mon = {'top': 0, 'left': 0, 'width': 480, 'height': 320}


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
    def __init__(self, port: str, ip="*"):
        self.observation_space = Box(0, 255, [mon['width'], mon['height'], 3])
        self.action_space = Box(-1, 1, [5, ])

        # init video capturing
        self.sct = mss()

        # start server
        subprocess.Popen([path_to_teeworlds + "teeworlds_srv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(0.2)

        # start client
        subprocess.Popen(["{}teeworlds".format(path_to_teeworlds), "gfx_screen_width {}".format(str(mon["width"])),
                          "gfx_screen_height {}".format(str(mon["height"])), "gfx_fullscreen 0", "gfx_borderless 1",
                          "cl_skip_start_menu 1", "connect localhost:8303"], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)

        move_window_to(0, 0)

        # create network context
        context = zmq.Context()
        self.socket = context.socket(zmq.PUSH)
        address = "tcp://{}:{}".format(ip, port)
        self.socket.bind(address)

        # waiting for everything to build up
        time.sleep(2)

    def step(self, action: Action):
        self.socket.send(action.to_bytes())
        observation = numpy.asarray(self.sct.grab(mon))
        # reward, done, info = self.socket.recv()
        return observation  # , reward, done, info

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


def move_window_to(x, y):
    time.sleep(0.5)
    subprocess.call(['xdotool', 'getactivewindow', 'windowmove', '--sync', str(x), str(y)])
