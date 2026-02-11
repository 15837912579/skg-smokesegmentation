"""
SKD-SegFormer Utility Functions
工具函数集合
"""
import os
import random
import logging
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(log_dir, phase='train'):
    """创建logger"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{phase}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create logger
    logger = logging.getLogger(phase)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def save_checkpoint(state, is_best, checkpoint_dir):
    """保存模型checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save latest
    checkpoint_path = os.path.join(checkpoint_dir, 'latest.pth')
    torch.save(state, checkpoint_path)
    
    # Save best
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_path)
    
    # Save periodic
    if state['epoch'] % 50 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'epoch_{state["epoch"]}.pth')
        torch.save(state, epoch_path)


def load_checkpoint(checkpoint_path):
    """加载checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(pred, target, threshold=0.5):
    """
    计算分割指标
    pred: (B, 1, H, W)
    target: (B, 1, H, W)
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    # Flatten
    pred_flat = pred.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # IoU (Intersection over Union)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, pred_flat, average='binary', zero_division=0
    )
    
    return {
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


class PolyLRScheduler:
    """Polynomial Learning Rate Scheduler"""
    def __init__(self, optimizer, max_epochs, warmup_epochs=0, 
                 warmup_lr=1e-6, min_lr=1e-6, power=0.9):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.power = power
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Warmup phase
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                 (self.current_epoch / self.warmup_epochs)
        else:
            # Polynomial decay
            progress = (self.current_epoch - self.warmup_epochs) / \
                      (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 (1 - progress) ** self.power
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr
        }
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']


import time  # Import for logger


if __name__ == '__main__':
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"AverageMeter test: avg={meter.avg}, count={meter.count}")
    
    # Test metrics calculation
    pred = torch.rand(2, 1, 256, 256)
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    metrics = calculate_metrics(pred, target)
    print(f"Metrics: {metrics}")
    
    print("\nUtility functions test completed!")
