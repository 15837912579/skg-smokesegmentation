"""
SKD-SegFormer Loss Functions
包含边界感知BCE损失和Dice损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class BoundaryAwareBCELoss(nn.Module):
    """边界感知加权BCE损失"""
    def __init__(self, boundary_weight=3.0, kernel_size=3):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size = kernel_size
        self.bce = nn.BCELoss(reduction='none')
    
    def extract_boundary(self, mask):
        """
        使用形态学操作提取边界
        mask: (B, 1, H, W)
        """
        B, _, H, W = mask.shape
        boundaries = []
        
        for i in range(B):
            m = mask[i, 0].cpu().numpy()
            m = (m * 255).astype(np.uint8)
            
            # Dilate and Erode
            kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
            dilated = cv2.dilate(m, kernel, iterations=1)
            eroded = cv2.erode(m, kernel, iterations=1)
            
            # Boundary = Dilate - Erode
            boundary = dilated - eroded
            boundary = torch.from_numpy(boundary).float() / 255.0
            boundaries.append(boundary)
        
        boundaries = torch.stack(boundaries, dim=0).unsqueeze(1)
        return boundaries.to(mask.device)
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W) - Predicted probability
        target: (B, 1, H, W) - Ground truth mask
        """
        # Extract boundaries
        boundaries = self.extract_boundary(target)
        
        # Create weight map
        weights = torch.ones_like(target)
        weights[boundaries > 0] = self.boundary_weight
        
        # Compute weighted BCE
        bce_loss = self.bce(pred, target)
        weighted_loss = (bce_loss * weights).mean()
        
        return weighted_loss


class DiceLoss(nn.Module):
    """Dice Loss"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CompositeLoss(nn.Module):
    """复合损失函数 = λ * BCE + (1-λ) * Dice"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, 
                 boundary_weight=3.0, kernel_size=3):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = BoundaryAwareBCELoss(boundary_weight, kernel_size)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        pred: (B, 1, H, W)
        target: (B, 1, H, W)
        """
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return total_loss, {'bce': bce.item(), 'dice': dice.item()}


def get_loss_function(cfg):
    """根据配置创建损失函数"""
    return CompositeLoss(
        bce_weight=cfg.TRAIN.LOSS.BCE_WEIGHT,
        dice_weight=cfg.TRAIN.LOSS.DICE_WEIGHT,
        boundary_weight=cfg.TRAIN.LOSS.BOUNDARY_WEIGHT,
        kernel_size=cfg.TRAIN.LOSS.KERNEL_SIZE
    )


if __name__ == '__main__':
    """Test loss functions"""
    print("Testing Loss Functions...")
    
    # Create dummy data
    pred = torch.rand(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    # Test BCE Loss
    bce_loss = BoundaryAwareBCELoss()
    loss_bce = bce_loss(pred, target)
    print(f"BCE Loss: {loss_bce.item():.4f}")
    
    # Test Dice Loss
    dice_loss = DiceLoss()
    loss_dice = dice_loss(pred, target)
    print(f"Dice Loss: {loss_dice.item():.4f}")
    
    # Test Composite Loss
    composite_loss = CompositeLoss()
    loss_total, loss_dict = composite_loss(pred, target)
    print(f"Total Loss: {loss_total.item():.4f}")
    print(f"Loss Dict: {loss_dict}")
    
    print("\nLoss functions test completed successfully!")
