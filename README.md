# Smart Cane

A Raspberry Pi 4 smart cane for blind and low-vision users, built solo at the SBU hackathon.
A camera runs an object detector, an ultrasonic sensor measures how far away the nearest thing is,
and the cane warns the user with a buzzer and short spoken alerts through a Bluetooth headset.
Detection and alerts run on the Pi, with no internet needed while it is running.

## How it works

| Part | Job |
|---|---|
| Pi camera + YOLOv8n (NCNN) | Says **what** is ahead and whether it is left, ahead or right |
| HC-SR04 ultrasonic sensor | Measures **how far** the nearest obstacle is |
| Buzzer | Beeps when something is closer than 1.2 m |
| Voice clips | Short ElevenLabs-generated mp3 clips, played with `mpg123` (falls back to `espeak`) |

Alert example: "person, ahead, very close".

## Hardware and wiring (BCM numbering)

| Part | Pi pin | GPIO |
|---|---|---|
| HC-SR04 VCC | Pin 2 (5 V) | |
| HC-SR04 TRIG | Pin 16 | GPIO 23 |
| HC-SR04 ECHO | Pin 18, **through a voltage divider** (1 kΩ + 1.8 kΩ) | GPIO 24 |
| Buzzer + | Pin 12 | GPIO 18 |
| Ground | Pin 14 | |

The ECHO pin outputs 5 V and Pi GPIO pins are 3.3 V, so the divider is required.

## Setup

On the Pi:
```
sudo apt install -y python3-picamera2 mpg123 espeak
pip install -r requirements-pi.txt --break-system-packages
```

Generate the voice clips on a laptop (needs an ElevenLabs API key in the environment):
```
pip install elevenlabs
export ELEVENLABS_API_KEY=...        # PowerShell: $env:ELEVENLABS_API_KEY="..."
python laptop/gen_voice_clips.py
scp -r clips <user>@<pi>:~/
```

Train the model (Colab) with `training/train.py`, then copy the exported `best_ncnn_model/` folder to the Pi.

Run on the Pi:
```
python3 pi/pi_cane_voice.py          # camera + ultrasonic + buzzer + voice
python3 pi/pi_cane_camera_only.py    # fallback without the sensor
python3 pi/pi_capture.py             # take your own training photos
```

Bluetooth headset: pair with `bluetoothctl` (`scan on`, `pair`, `trust`, `connect`).

## Model

YOLOv8n fine-tuned for 30 epochs on the Roboflow **SOD** sidewalk-obstacle dataset (22 classes),
exported to NCNN at 416 px. Validation mAP50 was 0.573, and it ran at about 5 frames per second on a Pi 4.
Check the dataset's own license on Roboflow before reusing it.

## Limitations

- The SOD dataset is outdoor sidewalk imagery. Indoors the detector often mislabels objects.
- The ultrasonic sensor has a wide beam and can pick up walls, tables, or the floor.
- The camera is fixed-focus, and distance comes from the sensor, not from the camera.
- This is a hackathon prototype. It is **not** a safety device and must not replace a white cane or a guide.

## Credits

Dataset: Roboflow SOD. Voice clips: ElevenLabs. Detector: Ultralytics YOLO.
