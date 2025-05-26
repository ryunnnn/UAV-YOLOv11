from ultralytics import YOLO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

weight = "/home/hnu3/mnt/yyk/yolov11/pruned.pt"

model = YOLO(weight)
# finetune设置为True
model.train(
    data='/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/datasets/myVisDrone.yaml',
    epochs=500,
    batch=16,
    imgsz=640,
    finetune=True,
    optimizer="SGD",
    device="7",
    workers=2,
    patience=50
)