import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    model_t = YOLO(r'/home/hnu3/mnt/yyk/yolov11/runs/detect/yolo11n_train/weights/best.pt')  # 此处填写教师模型的权重文件地址

    model_t.model.model[-1].set_Distillation = True  # 不用理会此处用于设置模型蒸馏

    model_s = YOLO(r'/home/hnu3/mnt/yyk/yolov11/pruned.pt')  # 学生文件的yaml文件 or 权重文件地址

    model_s.train(data=r'/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/datasets/myVisDrone.yaml',
                  # 将data后面替换你自己的数据集地址
                  epochs=500,
                  batch=8,
                  workers=2,
                  device="0",
                  optimizer="SGD",
                  amp=False,  # 如果出现训练损失为Nan可以关闭amp
                  patience=50,
                  model_t=model_t.model
                  )