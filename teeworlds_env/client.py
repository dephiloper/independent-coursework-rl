#! /usr/bin/env python3
import time

import zmq
import struct
import inputs


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


context = zmq.Context()

# Socket to talk to server
print("Connecting to hello world server…")
socket = context.socket(zmq.PUSH)
socket.bind("tcp://*:5555")


# while 1:
#     controls = Controls()
#     mouse_events = inputs.get_mouse()
#     for event in mouse_events:
#         print(event.ev_type, event.code, event.state)
#         #if event.code == "KEY_A" and event.state > 0:
#         #    controls.direction += -1
#         #if event.code == "KEY_A" and event.state > 0:
#         #    controls.direction += 1
#         #if event.code == "KEY_SPACE" and event.state > 0:
#         #    controls.jump = 1
#
#     print("before")
#     key_events = inputs.get_key()
#     if key_events:
#         for event in key_events:
#             if event.code == "KEY_A" and event.state > 0:
#                 controls.direction += -1
#             if event.code == "KEY_A" and event.state > 0:
#                 controls.direction += 1
#             if event.code == "KEY_SPACE" and event.state > 0:
#                 controls.jump = 1
#     print("after")

while True:
    controls = Controls()
    controls.mouse_x = -45
    controls.mouse_y = 12
    controls.direction = -1
    controls.jump = True
    controls.hook = False
    controls.shoot = True

    socket.send(controls.to_bytes())
    print("message send")
    time.sleep(1)
    #message = socket.recv()
    #print("Received reply \n %s" % message.decode("utf-8"))
