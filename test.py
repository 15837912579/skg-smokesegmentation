"""
SKD-SegFormer Testing Script
测试脚本，评估模型并保存结果
"""
import os
import sys
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# 导入本地模块
from config import cfg
from dataset import SmokeSegmentationDataset
from model import SKDSegFormer
from utils import create_logger, load_checkpoint, calculate_metrics, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Test SKD-SegFormer')
    
    # Data
    parser.add_argument('--data-root', type=str, default='./data',
                      help='Dataset root directory')
    parser.add_argument('--test-dir', type=str, default=None,
                      help='Test directory (default: data_root/test)')
    
    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results',
                      help='Output directory for results')
    parser.add_argument('--save-overlay', action='store_true',
                      help='Save prediction overlay on original image')
    parser.add_argument('--save-binary', action='store_true', default=True,
                      help='Save binary prediction masks')
    
    # System
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of data loading workers')
    
    args = parser.parse_args()
    return args


def denormalize_image(image, mean, std):
    """反归一化图像"""
    mean = torch.tensor(mean).view(3, 1, 1).to(image.device)
    std = torch.tensor(std).view(3, 1, 1).to(image.device)
    image = image * std + mean
    image = image.clamp(0, 1)
    return image


def save_results(image, pred, target, save_path, save_overlay=True):
    """保存预测结果"""
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        image = (image * 255).astype(np.uint8)
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy().squeeze()  # (H, W)
    
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy().squeeze()  # (H, W)
    
    # Save binary prediction
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    cv2.imwrite(save_path.replace('.png', '_pred.png'), pred_binary)
    
    # Save overlay
    if save_overlay:
        # Create colored overlay
        overlay = image.copy()
        pred_mask = pred > 0.5
        overlay[pred_mask] = overlay[pred_mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imwrite(save_path.replace('.png', '_overlay.png'), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # Save ground truth (if available)
    if target is not None:
        target_binary = (target > 0.5).astype(np.uint8) * 255
        cv2.imwrite(save_path.replace('.png', '_gt.png'), target_binary)


def test(model, test_loader, args, logger):
    """测试模型"""
    model.eval()
    
    # Metrics meters
    ious = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Testing on {len(test_loader.dataset)} images...")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for i, batch in enumerate(pbar):
            images = batch['image'].cuda(non_blocking=True)
            masks = batch['mask'].cuda(non_blocking=True)
            names = batch['name']
            
            # Forward
            outputs = model(images)
            # Resize output to match target size
            if outputs.shape[2:] != masks.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[2:],
                    mode="bilinear", align_corners=False
                )
            pred_binary = (outputs > 0.5).float()
            
            # Calculate metrics
            metrics = calculate_metrics(pred_binary, masks)
            
            # Update meters
            ious.update(metrics['iou'], images.size(0))
            precisions.update(metrics['precision'], images.size(0))
            recalls.update(metrics['recall'], images.size(0))
            f1s.update(metrics['f1'], images.size(0))
            
            # Save results
            for j in range(images.size(0)):
                # Denormalize image
                img = denormalize_image(images[j], cfg.DATASET.MEAN, cfg.DATASET.STD)
                
                # Save path
                save_path = os.path.join(args.output_dir, names[j])
                
                # Save
                save_results(
                    img, outputs[j], masks[j], save_path,
                    save_overlay=args.save_overlay
                )
            
            # Update progress bar
            pbar.set_postfix({
                'IoU': f'{ious.avg:.4f}',
                'F1': f'{f1s.avg:.4f}'
            })
    
    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("Test Results:")
    logger.info(f"  IoU:       {ious.avg:.4f}")
    logger.info(f"  Precision: {precisions.avg:.4f}")
    logger.info(f"  Recall:    {recalls.avg:.4f}")
    logger.info(f"  F1 Score:  {f1s.avg:.4f}")
    logger.info("=" * 60)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"IoU: {ious.avg:.4f}\n")
        f.write(f"Precision: {precisions.avg:.4f}\n")
        f.write(f"Recall: {recalls.avg:.4f}\n")
        f.write(f"F1 Score: {f1s.avg:.4f}\n")
    
    logger.info(f"Metrics saved to {metrics_file}")
    
    return {
        'iou': ious.avg,
        'precision': precisions.avg,
        'recall': recalls.avg,
        'f1': f1s.avg
    }


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = create_logger(args.output_dir, 'test')
    
    logger.info("=" * 80)
    logger.info("SKD-SegFormer Testing")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Output Directory: {args.output_dir}")
    
    # Set test directory
    if args.test_dir is None:
        args.test_dir = os.path.join(args.data_root, 'test')
    
    # Create test dataset
    logger.info(f"Loading test dataset from {args.test_dir}...")
    test_dataset = SmokeSegmentationDataset(
        root_dir=args.test_dir,
        split='test',
        img_size=cfg.DATASET.IMG_SIZE,
        mean=cfg.DATASET.MEAN,
        std=cfg.DATASET.STD,
        augmentation=None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    logger.info("Creating model...")
    model = SKDSegFormer(cfg).cuda()
    logger.info(f"Model parameters: {model.get_param_count() / 1e6:.2f}M")
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = load_checkpoint(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    if 'best_iou' in checkpoint:
        logger.info(f"Checkpoint best IoU: {checkpoint['best_iou']:.4f}")
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    
    # Test
    metrics = test(model, test_loader, args, logger)
    
    logger.info("\nTesting completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
