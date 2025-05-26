if __name__ == '__main__':
    from ultralytics import RTDETR
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    set_model = '/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/models/rt-detr/rtdetr-resnet18.yaml'  # 模型yaml文件路径
    #set_pre_pt = '/home/hnu3/mnt/yyk/yolov11/runs/detect/train4/weights/last.pt'  # 预训练模型路径
    set_data = '/home/hnu3/mnt/yyk/yolov11/ultralytics/cfg/datasets/myVisDrone.yaml'  # 数据路径

    model = RTDETR(set_model)
    model.train(data=set_data, batch=8, epochs=1000, workers=2, device="7", lr0=0.0001 ,lrf=0.001 ,optimizer='AdamW', momentum=0.9,weight_decay=0.0001,resume=True,name="rtdetr-r18")
    

