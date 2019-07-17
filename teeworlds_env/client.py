#! /usr/bin/env python3
import random
import struct
import subprocess
import time

import zmq

PATH_TO_TEEWORLDS = "/home/phil/Development/university/imi_master/2019sose/teeworlds/build/x86_64/debug/"


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


subprocess.Popen([PATH_TO_TEEWORLDS + "teeworlds_srv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
time.sleep(0.2)
subprocess.Popen([PATH_TO_TEEWORLDS + "teeworlds", "gfx_screen_width 480", "gfx_screen_height 320", "gfx_fullscreen 0",
                  "connect localhost:8303"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

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
    #print("message send")
    time.sleep(1)
    i+=1
