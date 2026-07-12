import os
import shutil
import random

# ====================== 配置参数 ======================
# 自动获取脚本所在的目录（也就是跌倒检测系统复现文件夹）
script_dir = os.path.dirname(os.path.abspath(__file__))
original_data_root = os.path.join(script_dir, "data")  # 原始data目录
output_root = os.path.join(script_dir, "pp_fall")      # 输出的数据集目录
train_ratio = 0.8                # 训练集占比
random_seed = 42                 # 随机种子，保证每次划分结果一致

# 类别名称（顺序对应类别ID：fallen=0, falling=1, standing=2）
class_list = ["fallen", "falling", "standing"]
# 支持的图片格式
image_suffix = [".jpg", ".jpeg", ".png", ".bmp"]
# =====================================================


def create_output_dirs():
    """创建输出目录结构"""
    dir_paths = [
        os.path.join(output_root, "images", "train"),
        os.path.join(output_root, "images", "val"),
        os.path.join(output_root, "labels", "train"),
        os.path.join(output_root, "labels", "val"),
    ]
    for path in dir_paths:
        os.makedirs(path, exist_ok=True)


def collect_all_images():
    """遍历所有类别文件夹，收集全部图片路径"""
    image_paths = []
    for cls_name in class_list:
        cls_dir = os.path.join(original_data_root, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"警告：类别文件夹 {cls_dir} 不存在，已跳过")
            continue
        for file_name in os.listdir(cls_dir):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in image_suffix:
                image_paths.append(os.path.join(cls_dir, file_name))
    return image_paths


def split_train_val(all_imgs):
    """按比例随机划分训练集、验证集"""
    random.seed(random_seed)
    random.shuffle(all_imgs)
    split_point = int(len(all_imgs) * train_ratio)
    train_set = all_imgs[:split_point]
    val_set = all_imgs[split_point:]
    print(f"总图片数量：{len(all_imgs)} | 训练集：{len(train_set)} | 验证集：{len(val_set)}")
    return train_set, val_set


def copy_files(img_list, dataset_type):
    """复制图片和对应标注文件到目标目录
    dataset_type: 'train' 或 'val'
    """
    for img_path in img_list:
        # 解析文件名
        img_dir, img_fullname = os.path.split(img_path)
        img_name_only = os.path.splitext(img_fullname)[0]

        # 目标路径
        target_img_dir = os.path.join(output_root, "images", dataset_type)
        target_label_dir = os.path.join(output_root, "labels", dataset_type)

        # 复制图片
        shutil.copy2(img_path, os.path.join(target_img_dir, img_fullname))

        # 复制对应标注txt
        label_file = img_name_only + ".txt"
        label_src_path = os.path.join(img_dir, label_file)
        if os.path.exists(label_src_path):
            shutil.copy2(label_src_path, os.path.join(target_label_dir, label_file))
        else:
            print(f"警告：图片 {img_fullname} 未找到对应标注文件 {label_file}")


def generate_yaml():
    """生成YOLO数据集配置yaml文件"""
    yaml_text = f"""# 跌倒检测数据集配置
path: {os.path.abspath(output_root)}  # 数据集根目录（绝对路径）
train: images/train                    # 训练集图片相对路径
val: images/val                        # 验证集图片相对路径

nc: {len(class_list)}                  # 类别总数
names: {class_list}                    # 类别名称列表
"""
    yaml_path = os.path.join(output_root, "fall_det.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f"\n数据集配置文件已生成：{yaml_path}")


if __name__ == "__main__":
    create_output_dirs()
    all_images = collect_all_images()

    if not all_images:
        print("错误：未找到任何图片，请检查 data 目录路径是否正确")
        exit()

    train_imgs, val_imgs = split_train_val(all_images)

    print("\n正在复制训练集文件...")
    copy_files(train_imgs, "train")
    print("正在复制验证集文件...")
    copy_files(val_imgs, "val")

    generate_yaml()
    print("\n✅ 数据集格式转换与划分完成！")
