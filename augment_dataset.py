import os
import cv2
import numpy as np
import albumentations as A

# ====================== 配置参数 ======================
script_dir = os.path.dirname(os.path.abspath(__file__))
train_img_dir = os.path.join(script_dir, "pp_fall", "images", "train")
train_label_dir = os.path.join(script_dir, "pp_fall", "labels", "train")

aug_per_image = 2
image_suffix = [".jpg", ".jpeg", ".png", ".bmp"]
# =====================================================

def get_aug_pipeline():
    """定义数据增强管线，适配跌倒检测场景"""
    return A.Compose([
        # 1. 几何变换
        A.HorizontalFlip(p=0.5),
        A.SmallestMaxSize(max_size=640, p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.4),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT),

        # 2. 光照与色彩增强
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),

        # 3. 噪声与模糊 (已修复新版参数要求)
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),  # 新版要求值在0-1之间
        A.MotionBlur(blur_limit=3, p=0.2),

        # 4. 随机遮挡 (已修复新版参数要求)
        A.CoarseDropout(
            num_holes_range=(1, 4),          # 替代 max_holes
            hole_height_range=(20, 40),      # 替代 max_height
            hole_width_range=(20, 40),       # 替代 max_width
            p=0.2
        )
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,  # 增强后框可见度低于30%则丢弃
        min_area=1,          # 过滤掉面积极小的无效框
        clip=True            # 自动将越界坐标(如极小负数)裁剪到 [0, 1] 范围内
    ))


def read_yolo_labels(label_path):
    """读取YOLO格式标注文件"""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if not line: continue
            parts = line.split()
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            bboxes.append([cx, cy, w, h])
            class_labels.append(cls_id)
    return bboxes, class_labels


def save_yolo_labels(label_path, bboxes, class_labels):
    """保存YOLO格式标注文件"""
    with open(label_path, 'w', encoding='utf-8') as f:
        for box, cls_id in zip(bboxes, class_labels):
            cx, cy, w, h = box
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def augment_dataset():
    transform = get_aug_pipeline()
    
    img_list = [f for f in os.listdir(train_img_dir) 
                if os.path.splitext(f)[1].lower() in image_suffix]
    
    total = len(img_list)
    print(f"找到训练图片 {total} 张，每张生成 {aug_per_image} 张增强样本")
    
    for idx, img_name in enumerate(img_list):
        img_path = os.path.join(train_img_dir, img_name)
        
        # 解决中文路径读取问题
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"跳过无法读取的图片：{img_name}")
            continue
        
        name_only = os.path.splitext(img_name)[0]
        label_path = os.path.join(train_label_dir, name_only + ".txt")
        bboxes, class_labels = read_yolo_labels(label_path)
        
        if not bboxes:
            print(f"跳过无标注图片：{img_name}")
            continue
        
        for i in range(aug_per_image):
            try:
                augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_labels = augmented['class_labels']
                
                if not aug_bboxes: continue
                
                aug_img_name = f"{name_only}_aug_{i}.jpg"
                aug_img_path = os.path.join(train_img_dir, aug_img_name)
                aug_label_name = f"{name_only}_aug_{i}.txt"
                aug_label_path = os.path.join(train_label_dir, aug_label_name)
                
                # 解决中文路径保存问题
                is_success, buffer = cv2.imencode(".jpg", aug_img)
                if is_success:
                    buffer.tofile(aug_img_path)
                
                save_yolo_labels(aug_label_path, aug_bboxes, aug_labels)
                
            except Exception as e:
                print(f"增强失败 {img_name} 第{i}张：{e}")
                continue
        
        if (idx + 1) % 50 == 0:
            print(f"已处理：{idx+1}/{total}")
    
    print("\n✅ 数据增强完成！增强样本已追加到训练集目录")
    print(f"训练集图片总量约：{total * (1 + aug_per_image)} 张")

if __name__ == "__main__":
    augment_dataset()