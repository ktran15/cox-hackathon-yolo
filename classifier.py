```python
import time
from collections import Counter

import numpy as np
import cv2

from gpiozero import Button
from picamera2 import Picamera2
from ultralytics import YOLO
from pi5neo import Pi5Neo

# -------------------------
# CONFIG
# -------------------------

MODEL_PATH = "best.onnx"
GPIO_PIN = 17

FRAME_COUNT = 3
FRAME_DELAY = 0.15
DISPLAY_TIME = 5

LED_COUNT = 105

# -------------------------
# LED SETUP
# -------------------------

strip = Pi5Neo("/dev/spidev0.0", LED_COUNT, 800)

# -------------------------
# CAMERA SETUP
# -------------------------

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
cam.start()

# -------------------------
# MODEL
# -------------------------

model = YOLO(MODEL_PATH)

# -------------------------
# GPIO SWITCH
# -------------------------

button = Button(GPIO_PIN)

# -------------------------
# COLORS
# -------------------------

GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
OFF = (0, 0, 0)

CLASS_MAP = {
    "shirt": (GREEN, BLUE),
    "pants": (GREEN, RED),
    "shoes": (YELLOW, PURPLE)
}

# -------------------------
# LED FUNCTIONS
# -------------------------

def clear():
    for i in range(LED_COUNT):
        strip.set_led_color(i, OFF)
    strip.update_strip()


def alternate(c1, c2):
    for i in range(LED_COUNT):
        strip.set_led_color(i, c1 if i % 2 == 0 else c2)
    strip.update_strip()

# -------------------------
# CLASSIFICATION
# -------------------------

def classify_3_frames():
    votes = []

    for _ in range(FRAME_COUNT):

        frame = cam.capture_array()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result = model.predict(frame, verbose=False)[0]

        label = result.names[int(result.probs.top1)]

        votes.append(label)

        time.sleep(FRAME_DELAY)

    return Counter(votes).most_common(1)[0][0]

# -------------------------
# MAIN LOOP
# -------------------------

clear()
print("System ready... waiting for switch")

while True:

    button.wait_for_press()

    print("Triggered!")

    cls = classify_3_frames()
    print("Detected:", cls)

    if cls in CLASS_MAP:
        c1, c2 = CLASS_MAP[cls]
        alternate(c1, c2)

        time.sleep(DISPLAY_TIME)

        clear()

    time.sleep(0.2)
```
