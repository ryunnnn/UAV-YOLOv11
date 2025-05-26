if __name__ == '__main__':
    from ultralytics import YOLO
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    set_model = '/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/models/v11/yolo11-iEMA.yaml'  # 模型yaml文件路径
    #set_model = "/home/hnu3/mnt/yyk/yolov11/yolo11m-visdrone.pt"
    set_pre_pt = '/home/hnu3/mnt/yyk/yolov11/runs/detect/train5/weights/last.pt'  # 预训练模型路径
    set_data = '/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/datasets/myVisDrone.yaml'  # 数据路径

    model = YOLO(set_pre_pt)
    model.train(data=set_data, batch=8, epochs=500, workers=2, device="7", amp=False, resume=True)
    #model.val(data=set_data)