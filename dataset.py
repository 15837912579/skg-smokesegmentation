"""
SKD-SegFormer Dataset
烟雾分割数据集加载和预处理
"""
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SmokeSegmentationDataset(Dataset):
    """
    烟雾分割数据集
    
    目录结构：
    data/
    ├── train/
    │   ├── images/  # RGB images (.jpg, .png)
    │   └── masks/   # Binary masks (.png)
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
    """
    
    def __init__(self, root_dir, split='train', img_size=(512, 512), 
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                 augmentation=None):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train', 'val', or 'test'
            img_size: (H, W) 图像大小
            mean: 归一化均值
            std: 归一化标准差
            augmentation: 数据增强配置
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.mean = mean
        self.std = std
        
        # 图像和掩码目录
        self.img_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        
        # 获取所有图像文件
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        print(f"[{split.upper()}] Found {len(self.img_files)} images in {root_dir}")
        
        # 定义数据增强
        if split == 'train' and augmentation:
            self.transform = self._get_train_transforms(augmentation)
        else:
            self.transform = self._get_val_transforms()
    
    def _get_train_transforms(self, aug_cfg):
        """训练集数据增强"""
        transforms = [
            A.Resize(self.img_size[0], self.img_size[1]),
        ]
        
        # 随机翻转
        if aug_cfg.RANDOM_FLIP:
            transforms.append(A.HorizontalFlip(p=0.5))
            transforms.append(A.VerticalFlip(p=0.3))
        
        # 随机旋转
        if aug_cfg.RANDOM_ROTATE:
            transforms.append(
                A.Rotate(limit=aug_cfg.ROTATE_LIMIT, p=0.5, 
                        border_mode=cv2.BORDER_CONSTANT, value=0)
            )
        
        # 颜色抖动
        if aug_cfg.COLOR_JITTER:
            transforms.append(
                A.ColorJitter(
                    brightness=aug_cfg.BRIGHTNESS,
                    contrast=aug_cfg.CONTRAST,
                    saturation=aug_cfg.SATURATION,
                    p=0.5
                )
            )
        
        # 其他增强
        transforms.extend([
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    def _get_val_transforms(self):
        """验证/测试集数据增强"""
        return A.Compose([
            A.Resize(self.img_size[0], self.img_size[1]),
            A.Normalize(mean=self.mean, std=self.std),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取掩码
        # 假设掩码文件名与图像相同，只是扩展名可能不同
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 如果找不到对应掩码，尝试其他扩展名
            for ext in ['.jpg', '.jpeg', '.bmp']:
                mask_path_alt = os.path.join(self.mask_dir, 
                                            os.path.splitext(img_name)[0] + ext)
                if os.path.exists(mask_path_alt):
                    mask = cv2.imread(mask_path_alt, cv2.IMREAD_GRAYSCALE)
                    break
            else:
                raise FileNotFoundError(f"Mask not found for {img_name}")
        
        # 二值化掩码 (确保只有0和1)
        mask = (mask > 127).astype(np.uint8)
        
        # 应用数据增强
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # 转换掩码为浮点型并增加通道维度
        mask = mask.unsqueeze(0).float()
        
        return {
            'image': image,
            'mask': mask,
            'name': img_name
        }


def create_dataloaders(cfg):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        cfg: 配置对象
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 训练集
    train_dataset = SmokeSegmentationDataset(
        root_dir=cfg.TRAIN_DIR,
        split='train',
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        augmentation=cfg.TRAIN.AUGMENTATION
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    # 验证集
    val_dataset = SmokeSegmentationDataset(
        root_dir=cfg.VAL_DIR,
        split='val',
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        augmentation=None
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.VAL.NUM_WORKERS,
        pin_memory=True
    )
    
    # 测试集
    test_dataset = SmokeSegmentationDataset(
        root_dir=cfg.TEST_DIR,
        split='test',
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        augmentation=None
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )
    
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print(f"  Train: {len(train_dataset)} images")
    print(f"  Val:   {len(val_dataset)} images")
    print(f"  Test:  {len(test_dataset)} images")
    print("=" * 60 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    """测试数据集加载"""
    import sys
    sys.path.append('.')
    from config import cfg
    
    print("Testing SmokeSegmentationDataset...")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    
    # 测试训练集
    print("\n[Testing Train Loader]")
    for batch in train_loader:
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Mask shape: {batch['mask'].shape}")
        print(f"  Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
        print(f"  Mask unique values: {torch.unique(batch['mask'])}")
        break
    
    print("\nDataset test completed successfully!")
