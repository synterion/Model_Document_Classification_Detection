from ultralytics import YOLO

model = YOLO('yolo26n.yaml')
model.train(data="config.yaml", epochs=100)
model.val(split='test')

