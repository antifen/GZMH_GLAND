import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torch
from skimage import measure, morphology
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# 检查文件是否为图像
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# 图像加载函数
def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')
    return img

def extract_glands_from_label(label_img):
    """
    从彩色标签图中提取所有腺体
    背景: 黑色 (0, 0, 0)
    良性腺体: 绿色 (0, 255, 0)
    恶性腺体: 红色 (255, 0, 0)
    
    返回: [(gland_patch, label), ...]
    label: 0=良性, 1=恶性
    """
    label_array = np.array(label_img)
    
    # 分离良性（绿色）和恶性（红色）区域
    # 绿色通道高，红色和蓝色通道低 -> 良性
    benign_mask = (label_array[:, :, 1] > 200) & (label_array[:, :, 0] < 50) & (label_array[:, :, 2] < 50)
    # 红色通道高，绿色和蓝色通道低 -> 恶性
    malignant_mask = (label_array[:, :, 0] > 200) & (label_array[:, :, 1] < 50) & (label_array[:, :, 2] < 50)
    
    glands = []
    
    # 提取良性腺体
    if np.any(benign_mask):
        # benign_labeled = measure.label(benign_mask, connectivity=2)
        # benign_labeled = morphology.remove_small_objects(benign_labeled, min_size=10)
        # benign_labeled = measure.label(benign_labeled, connectivity=2)
        # 先移除小对象（使用布尔数组）
        benign_mask_cleaned = morphology.remove_small_objects(benign_mask, min_size=10)
        # 然后进行连通域标记
        benign_labeled = measure.label(benign_mask_cleaned, connectivity=2)
        
        for region_id in range(1, benign_labeled.max() + 1):
            region_mask = benign_labeled == region_id
            if np.sum(region_mask) < 10:  # 过滤太小的区域
                continue
            
            # 计算边界框
            coords = np.argwhere(region_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 提取区域（包含一些padding）
            padding = 5
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(label_array.shape[0], y_max + padding)
            x_max = min(label_array.shape[1], x_max + padding)
            
            glands.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'label': 0,  # 良性
                'mask': region_mask[y_min:y_max, x_min:x_max]
            })
    
    # 提取恶性腺体
    if np.any(malignant_mask):
        # malignant_labeled = measure.label(malignant_mask, connectivity=2)
        # malignant_labeled = morphology.remove_small_objects(malignant_labeled, min_size=10)
        # malignant_labeled = measure.label(malignant_labeled, connectivity=2)
        # 先移除小对象（使用布尔数组）
        malignant_mask_cleaned = morphology.remove_small_objects(malignant_mask, min_size=10)
        # 然后进行连通域标记
        malignant_labeled = measure.label(malignant_mask_cleaned, connectivity=2)
        
        for region_id in range(1, malignant_labeled.max() + 1):
            region_mask = malignant_labeled == region_id
            if np.sum(region_mask) < 10:  # 过滤太小的区域
                continue
            
            # 计算边界框
            coords = np.argwhere(region_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # 提取区域（包含一些padding）
            padding = 5
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(label_array.shape[0], y_max + padding)
            x_max = min(label_array.shape[1], x_max + padding)
            
            glands.append({
                'bbox': (x_min, y_min, x_max, y_max),
                'label': 1,  # 恶性
                'mask': region_mask[y_min:y_max, x_min:x_max]
            })
    
    return glands

class ClassificationDataFolder(data.Dataset):
    """
    分类数据集：从彩色标签图中提取腺体并进行分类
    """
    def __init__(self, img_dir, label_dir, data_transform=None, target_size=416, loader=img_loader):
        super(ClassificationDataFolder, self).__init__()
        
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.data_transform = data_transform
        self.target_size = target_size
        self.loader = loader
        
        # 获取所有图像文件
        img_files = [f for f in os.listdir(img_dir) if is_image_file(f)]
        
        # 提取所有腺体
        self.gland_list = []
        for img_file in img_files:
            name = os.path.splitext(img_file)[0]
            img_path = os.path.join(img_dir, img_file)
            
            # 查找对应的标签文件
            label_path = None
            for ext in ['_mask_color.png', '_label.png', '_mask.png']:
                potential_path = os.path.join(label_dir, name + ext)
                if os.path.exists(potential_path):
                    label_path = potential_path
                    break
            
            if label_path is None:
                print(f"Warning: No label found for {img_file}")
                continue
            
            # 加载图像和标签
            img = self.loader(img_path, 3)
            label_img = self.loader(label_path, 3)
            
            # 提取腺体
            glands = extract_glands_from_label(label_img)
            
            for gland_info in glands:
                self.gland_list.append({
                    'img_path': img_path,
                    'bbox': gland_info['bbox'],
                    'label': gland_info['label'],
                    'name': name
                })
        
        print(f"Total glands extracted: {len(self.gland_list)}")
        print(f"Benign: {sum(1 for g in self.gland_list if g['label'] == 0)}, "
              f"Malignant: {sum(1 for g in self.gland_list if g['label'] == 1)}")
    
    def __getitem__(self, index):
        gland_info = self.gland_list[index]
        
        # 加载原始图像
        img = self.loader(gland_info['img_path'], 3)
        img_array = np.array(img)
        
        # 提取腺体区域的矩形框
        x_min, y_min, x_max, y_max = gland_info['bbox']
        gland_patch = img_array[y_min:y_max, x_min:x_max, :]

        # 转换为PIL Image
        gland_patch = Image.fromarray(gland_patch)
        
        # Resize到目标尺寸
        # gland_patch = gland_patch.resize((self.target_size, self.target_size), Image.BILINEAR)


        # 原始尺寸
        w, h = gland_patch.size
        
        # 1. 计算缩放比例：让最长边等于 target_size
        scale = self.target_size / max(w, h)
        
        # 2. 计算新的宽和高（保持长宽比）
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 3. 进行缩放
        gland_patch = gland_patch.resize((new_w, new_h), Image.BILINEAR)
        
        # 4. 创建正方形画布（Padding）
        # 病理图像背景通常是白色的，所以这里填充 (255, 255, 255)
        # 如果你的预处理去除了背景变黑了，这里可以改为 (0, 0, 0)
        final_img = Image.new('RGB', (self.target_size, self.target_size), (255, 255, 255))
        # final_img = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        
        # 5. 计算粘贴位置（居中）
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        
        # 6. 将缩放后的图粘贴到画布中心
        final_img.paste(gland_patch, (paste_x, paste_y))
        
        # 更新变量名为 final_img 以供后续使用
        gland_patch = final_img       
        
        # 应用数据增强
        if self.data_transform is not None:
            gland_patch = self.data_transform(gland_patch)
        else:
            # 默认转换为tensor
            gland_patch = torch.from_numpy(np.array(gland_patch).transpose((2, 0, 1))).float() / 255.0
        
        label = torch.tensor(gland_info['label'], dtype=torch.long)
        
        return gland_patch, label
    
    def __len__(self):
        return len(self.gland_list)