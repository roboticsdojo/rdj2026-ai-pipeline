import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("nikita-kamau").project("rdj2026")
dataset = project.version(1).download("yolov8")  # adjust version number and format as needed