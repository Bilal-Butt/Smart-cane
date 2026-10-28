"""Smart cane: camera detector + ultrasonic distance -> buzzer + spoken alerts.

Wiring (BCM pin numbers):
  HC-SR04 TRIG -> GPIO 23
  HC-SR04 ECHO -> GPIO 24 THROUGH a voltage divider (1k + 2k ohm). Echo is 5V, Pi pins are 3.3V!
  HC-SR04 VCC  -> 5V, GND -> GND
  Buzzer / vibration motor (via a transistor for the motor) -> GPIO 18
  Pi Camera -> CSI port

Setup on the Pi:
  sudo apt update && sudo apt install -y python3-picamera2 espeak
  pip install ultralytics ncnn gpiozero --break-system-packages
Copy the exported folder best_ncnn_model/ next to this file, then:  python3 pi_cane.py
"""
import subprocess
import time

from gpiozero import Buzzer, DistanceSensor
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "best_ncnn_model"
CONF = 0.45              # detection confidence threshold
DANGER_M = 1.2           # warn when something is closer than this (meters)
SPEAK_COOLDOWN = 2.5     # seconds between spoken alerts

sensor = DistanceSensor(echo=24, trigger=23, max_distance=3)
buzzer = Buzzer(18)
model = YOLO(MODEL_PATH, task="detect")

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(1)

last_spoken = 0.0


def speak(text):
    print(text)  # always show the alert in the terminal too
    try:
        # espeak -> paplay sends sound to the default audio sink,
        # which is the Bluetooth headset once it is paired and connected
        subprocess.Popen(
            f'espeak -s 160 "{text}" --stdout | paplay',
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


print("Smart cane running. Ctrl+C to stop.")
try:
    while True:
        frame = cam.capture_array()  # BGR layout, which is what YOLO expects
        result = model.predict(frame, imgsz=416, conf=CONF, verbose=False)[0]
        dist = sensor.distance  # meters, max 3
        now = time.time()

        # pick the largest detected box (usually the nearest object)
        target = None
        best_area = 0
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                target = (result.names[int(box.cls[0])], side((x1 + x2) / 2, frame.shape[1]))

        if dist < DANGER_M:
            buzzer.beep(on_time=0.08, off_time=0.08, n=3, background=True)
            if now - last_spoken > SPEAK_COOLDOWN:
                if target:
                    speak(f"{target[0]} {target[1]}, {dist:.1f} meters")
                else:
                    speak(f"Obstacle {dist:.1f} meters")
                last_spoken = now
        elif target and now - last_spoken > SPEAK_COOLDOWN * 2:
            # something seen by the camera but still far: soft heads-up
            speak(f"{target[0]} {target[1]}")
            last_spoken = now
finally:
    buzzer.off()
    cam.stop()
