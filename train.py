from ultralytics import YOLO

# 加载 YOLOv5n 预训练模型（轻量级，适合大作业快速验证）
model = YOLO('yolov5n.pt')  

# 开始训练
model.train(
    data='pp_fall/fall_det.yaml',
    epochs=100,              # 训练轮数，可根据效果调整
    batch=16,                # 批次大小，如果显存爆掉请改为 8
    imgsz=640,               # 输入图像尺寸
    project='runs/detect',   # 结果保存路径
    name='fall_det_train',   # 本次训练名称
    # 以下参数开启数据增强
    mosaic=1.0,              
    mixup=0.1,               
    degrees=10.0,            
    scale=0.5,               
    fliplr=0.5               
)