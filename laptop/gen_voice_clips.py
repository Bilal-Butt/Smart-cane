"""Generate ElevenLabs voice clips for the smart cane (run on your LAPTOP, not the Pi).

Setup:
  pip install elevenlabs
  Windows PowerShell:  $env:ELEVENLABS_API_KEY="your_key_here"
Run:
  python gen_voice_clips.py
Then copy the clips folder to the Pi:
  scp -r clips bilal@cane.local:~/
"""
import os

from elevenlabs.client import ElevenLabs

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
if not API_KEY:
    raise SystemExit("Set ELEVENLABS_API_KEY first (see the top of this file).")

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"   # "George" from the ElevenLabs docs; any voice id works
MODEL_ID = "eleven_flash_v2_5"      # fast, low-latency model
OUT_DIR = "clips"

CLASSES = [
    "bench", "bicycle", "bus", "bus_stop", "cane", "car", "curb", "dog",
    "fire_hydrant", "motorcycle", "person", "pole", "spherical_roadblock",
    "stairs", "stop_sign", "street_light", "traffic_light", "train", "tree",
    "truck", "warning_column", "waste_container",
    # extra classes from the stock COCO model
    "chair", "couch", "potted_plant", "dining_table",
]
EXTRA = {
    "left": "left",
    "ahead": "ahead",
    "right": "right",
    "close": "close",
    "very_close": "very close",
    "obstacle": "obstacle ahead",
}

phrases = {name: name.replace("_", " ") for name in CLASSES}
phrases.update(EXTRA)

os.makedirs(OUT_DIR, exist_ok=True)
client = ElevenLabs(api_key=API_KEY)

for key, text in phrases.items():
    path = os.path.join(OUT_DIR, f"{key}.mp3")
    if os.path.exists(path):
        continue
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="mp3_44100_128",
    )
    with open(path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print("saved", path)

print("Done:", len(phrases), "clips in", OUT_DIR)
