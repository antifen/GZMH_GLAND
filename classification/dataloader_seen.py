import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from classification_data_folder import ClassificationDataFolder

# 和 train/val/test 保持一致的 transform（如果你有改，这里也一起改）
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # transforms.RandomRotation(degrees=90),
        transforms.RandomAffine(
            degrees=180, 
            fill=0  # <--- 重要！保持背景为白色
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # 加大明暗对比变化，模拟染色深浅；减小色相变化，防止颜色失真
        # transforms.ColorJitter(brightness=0.35, contrast=0.5, saturation=0.4, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 1. 构建数据集（示例：用训练集的 img_dir / label_dir）
img_dir = "./data/ganzhou_part1/img/test"      # 改成你自己配置里的路径
label_dir = "./data/ganzhou_part1/mask_color/test"  # 或 label 目录

dataset = ClassificationDataFolder(
    img_dir=img_dir,
    label_dir=label_dir,
    data_transform=data_transform,
    target_size=512
)

# 2. 创建输出文件夹
save_dir = "./debug_preprocessed_patches"
os.makedirs(save_dir, exist_ok=True)

# 3. 反归一化函数（把 tensor 变回看得懂的图像）
inv_mean = torch.tensor(mean).view(3, 1, 1)
inv_std = torch.tensor(std).view(3, 1, 1)

def denormalize(t):
    return t * inv_std + inv_mean

# 4. 遍历前 N 个样本，保存预处理后的图像
N = 50  # 想看多少个就改多少
for idx in range(min(N, len(dataset))):
    img_tensor, label = dataset[idx]  # img_tensor: [C,H,W]，已经做完所有预处理

    # 反归一化到 [0,1] 范围
    img_vis = denormalize(img_tensor.clone())
    img_vis = torch.clamp(img_vis, 0.0, 1.0)

    save_path = os.path.join(save_dir, f"gland_{idx:04d}_label_{int(label)}.png")
    save_image(img_vis, save_path)

    print(f"Saved: {save_path}")

print("Done. Check folder:", save_dir)