import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/hnu3/mnt/yyk/yolov11/runs/detect/yolov11-kd/weights/best.pt') # select your model.pt path
    model.predict(source='/home/hnu3/mnt/yyk/yolov11/datasets/VisDrone/VisDrone2019-DET-test-dev/images/0000006_00159_d_0000001.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                )