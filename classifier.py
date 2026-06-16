import time
from collections import Counter
import cv2
import numpy as np
import onnxruntime as ort
from gpiozero import Button
from picamera2 import Picamera2
from pi5neo import Pi5Neo

# =====================
# CONFIG
# =====================
MODEL_PATH = "best.onnx"
GPIO_PIN = 17
FRAME_COUNT = 3
FRAME_DELAY = 0.15
DISPLAY_SECONDS = 5
LED_COUNT = 105
CONF_THRESHOLD = 0.8
INPUT_SIZE = 224

LABELS = ["pants", "shirt", "shoes"]   # Roboflow order: 0=pants, 1=shirt, 2=shoes

CLASS_COLORS = {
    "shirt": ((0, 255, 0), (0, 0, 255)),     # green / blue
    "pants": ((0, 255, 0), (255, 0, 0)),     # green / red
    "shoes": ((255, 255, 0), (255, 0, 255)), # yellow / purple
}

# =====================
# LED (pi5neo — same as piano project)
# =====================
strip = Pi5Neo("/dev/spidev0.0", LED_COUNT, 800)

def clear_leds():
    strip.fill_strip(0, 0, 0)
    strip.update_strip()

def set_pattern(c1, c2):
    for i in range(LED_COUNT):
        r, g, b = c1 if i % 2 == 0 else c2
        strip.set_led_color(i, r, g, b)
    strip.update_strip()

def set_solid(r, g, b):
    strip.fill_strip(r, g, b)
    strip.update_strip()

# =====================
# MODEL (onnxruntime only)
# =====================
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))     # HWC -> CHW (NCHW)
    img = np.expand_dims(img, axis=0)
    return img

def predict(frame):
    inp = preprocess(frame)
    output = session.run(None, {input_name: inp})[0]
    probs = output[0]
    if probs.max() > 1.0 or probs.min() < 0.0:   # logits -> probabilities
        probs = softmax(probs)
    cls = int(np.argmax(probs))
    return LABELS[cls], float(probs[cls])

# =====================
# CAMERA
# =====================
camera = Picamera2()
camera.configure(
    camera.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
camera.start()
time.sleep(2)

# =====================
# VOTING
# =====================
def classify():
    votes = []
    for _ in range(FRAME_COUNT):
        frame = camera.capture_array()
        label, conf = predict(frame)
        print(f"Frame: {label} ({conf:.2f})")
        if conf >= CONF_THRESHOLD:
            votes.append(label)
        time.sleep(FRAME_DELAY)
    return Counter(votes).most_common(1)[0][0] if votes else None

# =====================
# GPIO + LOOP
# =====================
button = Button(GPIO_PIN, bounce_time=0.05)
print("Ready.")
clear_leds()

while True:
    button.wait_for_press()
    print("Switch triggered")
    result = classify()
    print("Final:", result)

    if result in CLASS_COLORS:
        c1, c2 = CLASS_COLORS[result]
        set_pattern(c1, c2)
    else:
        print("Not confident — red")
        set_solid(255, 0, 0)   # unsure / low confidence

    time.sleep(DISPLAY_SECONDS)
    clear_leds()
    while button.is_pressed:
        time.sleep(0.1)