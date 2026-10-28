"""Smart cane, camera-only version (no ultrasonic sensor, no buzzer).

Distance is estimated from how much of the frame an object fills:
  tall box = close, small box = far. It is a rough guide, not a measurement.

Run on the Pi:  python3 pi_cane_camera_only.py
Stop with Ctrl+C.
Output: printed alerts in the terminal, plus spoken alerts if audio works.
"""
import subprocess
import time

from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best_ncnn_model"
CONF = 0.45              # detection confidence threshold
VERY_CLOSE = 0.60        # box height / frame height above this = very close
CLOSE = 0.35             # above this = close
SPEAK_COOLDOWN = 2.5     # seconds between spoken alerts
SPEAK = True             # set False for silent (terminal only)

model = YOLO(MODEL_PATH, task="detect")
cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(1)

last_spoken = 0.0


def speak(text):
    if not SPEAK:
        return
    try:
        subprocess.Popen(
            f'espeak "{text}" --stdout | paplay',
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def side(x_center, width):
    ratio = x_center / width
    if ratio < 0.33:
        return "left"
    if ratio > 0.67:
        return "right"
    return "ahead"


print("Camera-only cane running. Ctrl+C to stop.")
try:
    while True:
        frame = cam.capture_array()  # BGR layout, which is what YOLO expects
        h, w = frame.shape[:2]
        result = model.predict(frame, imgsz=416, conf=CONF, verbose=False)[0]

        target = None
        best_frac = 0.0
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            frac = (y2 - y1) / h
            if frac > best_frac:
                best_frac = frac
                target = (result.names[int(box.cls[0])], side((x1 + x2) / 2, w), frac)

        now = time.time()
        if target and target[2] >= CLOSE and now - last_spoken > SPEAK_COOLDOWN:
            name, where, frac = target
            level = "very close" if frac >= VERY_CLOSE else "close"
            msg = f"{name} {where}, {level}"
            print(msg)
            speak(msg)
            last_spoken = now
finally:
    cam.stop()
