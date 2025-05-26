if __name__ == '__main__':
    from ultralytics import YOLO
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    set_model = '/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/models/v10/yolov10n.yaml'  # 模型yaml文件路径
    #set_pre_pt = '/home/hnu3/mnt/yyk/yolov11/runs/detect/yolo11_kd_train1/weights/best.pt'  # 预训练模型路径
    set_data = '/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/datasets/myVisDrone.yaml'  # 数据路径

    model = YOLO(set_model)
    model.train(data=set_data, batch=8, epochs=500, workers=1, device="5", patience=50)