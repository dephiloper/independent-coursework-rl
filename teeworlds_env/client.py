#! /usr/bin/env python3
import random
import struct
import subprocess
import time

import cv2
import numpy
import zmq
from mss import mss

mon = {'top': 0, 'left': 0, 'width': 320, 'height': 320}

with open('config.txt', 'r') as f:
    l = f.readline()
    path_to_teeworlds = l.strip()


class Controls:
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


def move_window_to(x, y):
    time.sleep(0.5)
    subprocess.call(['xdotool', 'getactivewindow', 'windowmove', '--sync', str(x), str(y)])


sct = mss()

# start server
subprocess.Popen([path_to_teeworlds + "teeworlds_srv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(0.2)

# for i in range(4):
# start client1
subprocess.Popen([path_to_teeworlds + "teeworlds", "gfx_screen_width {0}".format(str(mon["width"])),
                  "gfx_screen_height " + str(mon["height"]), "gfx_fullscreen 0", "gfx_borderless 1",
                  "cl_skip_start_menu 1",
                  "connect localhost:8303"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
move_window_to(0, 0)

context = zmq.Context()

# Socket to talk to server
print("waiting for connections")
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5555")

controls = Controls()
controls.mouse_x = 1
controls.mouse_y = 1
i = 0
while True:
    controls.mouse_x = random.randrange(-200, 200)
    controls.mouse_y = random.randrange(-200, 200)
    controls.direction = 1 if controls.direction < 0 else -1
    controls.jump = i % 3
    controls.hook = False
    controls.shoot = i % 5

    socket.send(controls.to_bytes())
    i += 1
    img = numpy.asarray(sct.grab(mon))
    cv2.waitKey()
