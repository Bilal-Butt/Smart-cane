"""Capture training photos with the cane's own Pi camera.

Run on the Pi:  python3 pi_capture.py
  Enter      -> save 1 photo
  b + Enter  -> save a burst of 5 photos (0.6 s apart), good for moving the camera slightly
  q + Enter  -> quit
Photos go to ~/photos/ . Copy them to the laptop with (run on the laptop):
  scp -r bilal@cane.local:~/photos .
"""
import os
import time

from picamera2 import Picamera2

OUT_DIR = "photos"
os.makedirs(OUT_DIR, exist_ok=True)

cam = Picamera2()
cam.configure(cam.create_still_configuration(main={"size": (640, 480)}))
cam.start()
time.sleep(1)

count = len(os.listdir(OUT_DIR))


def snap():
    global count
    count += 1
    path = os.path.join(OUT_DIR, f"cane_{count:04d}.jpg")
    cam.capture_file(path)
    print("saved", path)


print("Enter = photo, b = burst of 5, q = quit")
try:
    while True:
        cmd = input("> ").strip().lower()
        if cmd == "q":
            break
        if cmd == "b":
            for _ in range(5):
                snap()
                time.sleep(0.6)
        else:
            snap()
finally:
    cam.stop()
    print("Total photos:", count)
