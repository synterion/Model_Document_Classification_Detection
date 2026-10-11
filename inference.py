from ultralytics import YOLO
import os

INFERENCE_PATH = "dataset_inference/Final/"

images = os.listdir(INFERENCE_PATH)
model = YOLO("best.pt")

for i in images :
    image = INFERENCE_PATH+i
    results = model(image)

    for r in results:
        print(r.boxes.cls)

    



