"""Smart cane with ElevenLabs voice clips: camera detector + ultrasonic distance -> buzzer + spoken alerts.

Same wiring as pi_cane.py (BCM pins): TRIG 23, ECHO 24 (through the divider), buzzer 18.

Setup on the Pi (once):
  sudo apt install -y mpg123
Copy the clips folder (made by gen_voice_clips.py on the laptop) next to this file:
  scp -r clips bilal@cane.local:~/
Run:  python3 pi_cane_voice.py

If a clip is missing or mpg123 fails, it falls back to espeak, so alerts never go silent.
"""
import os
import statistics
import subprocess
import time
from collections import deque

from gpiozero import Buzzer, DistanceSensor
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best_ncnn_model"   # sidewalk model. For the stock COCO model use "yolov8n.pt" or "yolov8n_ncnn_model"
# Only announce these classes (None = announce everything). Handy with the COCO model:
ALLOWED = None
# ALLOWED = {"person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light",
#            "fire hydrant", "stop sign", "bench", "dog", "chair", "couch",
#            "potted plant", "dining table"}
CLIPS_DIR = "clips"
CONF = 0.60              # detection confidence threshold (higher = fewer false alarms)
CONFIRM_FRAMES = 2       # same object must be seen this many frames in a row
MIN_VALID_M = 0.04       # readings below this are sensor noise, ignore them
DEBUG = True             # print distance + detection every loop
DANGER_M = 1.2           # warn when something is closer than this (meters)
VERY_CLOSE_M = 0.6       # say "very close" below this
SPEAK_COOLDOWN = 2.5     # seconds between spoken alerts
PLAYER = ["mpg123", "-q"]  # if the headset stays silent, try ["mpg123", "-q", "-o", "pulse"]

sensor = DistanceSensor(echo=24, trigger=23, max_distance=3)
buzzer = Buzzer(18)
model = YOLO(MODEL_PATH, task="detect")

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(1)

last_spoken = 0.0
player_proc = None
recent_dist = deque(maxlen=3)   # median of the last 3 readings removes one-off spikes
streak_name = None
streak_count = 0


def is_playing():
    return player_proc is not None and player_proc.poll() is None


def speak_clips(keys, fallback_text):
    """Play clips/<key>.mp3 for each key in order. Falls back to espeak if any is missing."""
    global player_proc
    print(fallback_text)  # always show the alert in the terminal too
    if is_playing():
        return  # don't talk over the previous alert
    paths = [os.path.join(CLIPS_DIR, f"{k.replace(' ', '_')}.mp3") for k in keys]
    try:
        if all(os.path.exists(p) for p in paths):
            player_proc = subprocess.Popen(
                PLAYER + paths, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            player_proc = subprocess.Popen(
                f'espeak -s 160 "{fallback_text}" --stdout | paplay',
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def side(x_center, width):
    ratio = x_center / width
    if ratio < 0.33:
        return "left"
    if ratio > 0.67:
        return "right"
    return "ahead"


print("Smart cane (voice clips) running. Ctrl+C to stop.")
try:
    while True:
        frame = cam.capture_array()  # BGR layout, which is what YOLO expects
        result = model.predict(frame, imgsz=416, conf=CONF, verbose=False)[0]
        raw = sensor.distance  # meters, max 3
        if raw >= MIN_VALID_M:
            recent_dist.append(raw)
        dist = statistics.median(recent_dist) if recent_dist else 3.0
        now = time.time()

        # pick the largest detected box (usually the nearest object)
        target = None
        best_area = 0
        for box in result.boxes:
            if ALLOWED and result.names[int(box.cls[0])] not in ALLOWED:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                target = (result.names[int(box.cls[0])], side((x1 + x2) / 2, frame.shape[1]))

        # only trust a detection seen in several frames in a row
        if target and target[0] == streak_name:
            streak_count += 1
        else:
            streak_name = target[0] if target else None
            streak_count = 1 if target else 0
        if streak_count < CONFIRM_FRAMES:
            target = None

        if DEBUG:
            print(f"dist={dist:.2f}m raw={raw:.2f} seen={target[0] + ' ' + target[1] if target else '-'}")

        if dist < DANGER_M:
            buzzer.beep(on_time=0.08, off_time=0.08, n=3, background=True)
            if now - last_spoken > SPEAK_COOLDOWN:
                level = "very_close" if dist < VERY_CLOSE_M else "close"
                level_text = level.replace("_", " ")
                if target:
                    name, where = target
                    speak_clips([name, where, level],
                                f"{name.replace('_', ' ')} {where}, {level_text}")
                else:
                    speak_clips(["obstacle", level], f"obstacle ahead, {level_text}")
                last_spoken = now
        elif target and now - last_spoken > SPEAK_COOLDOWN * 2:
            # something seen by the camera but still far: soft heads-up
            name, where = target
            speak_clips([name, where], f"{name.replace('_', ' ')} {where}")
            last_spoken = now
finally:
    buzzer.off()
    cam.stop()
