"""Train YOLO on the Roboflow SOD sidewalk-obstacle dataset (run in Google Colab, T4 GPU).

Set your Roboflow key as an environment variable. Never commit it:
  import os; os.environ["ROBOFLOW_API_KEY"] = "..."   # in a Colab cell, not in this file
Install:  pip install ultralytics roboflow
"""
import os

from roboflow import Roboflow
from ultralytics import YOLO

WORKSPACE = "bilal-butt-klzzc"
PROJECT = "sod-enect-znl81"
VERSION = 1            # change to the version number you created in Roboflow
BASE_MODEL = "yolov8n.pt"   # or "yolo26n.pt" if your Ultralytics version supports it

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov8")

model = YOLO(BASE_MODEL)
model.train(data=f"{dataset.location}/data.yaml", epochs=30, imgsz=416, batch=32, patience=8)

# Export for the Raspberry Pi (CPU) as NCNN
YOLO("runs/detect/train/weights/best.pt").export(format="ncnn", imgsz=416)
