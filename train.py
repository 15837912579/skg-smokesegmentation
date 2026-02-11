"""
SKD-SegFormer Training Script
完整的训练流程
"""
import os
import sys
import argparse
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# 导入本地模块
from config import cfg, update_config
from dataset import create_dataloaders
from model import SKDSegFormer
from losses import get_loss_function
from utils import (
    set_seed, create_logger, save_checkpoint, load_checkpoint,
    AverageMeter, calculate_metrics, PolyLRScheduler
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SKD-SegFormer')
    
    # Data
    parser.add_argument('--data-root', type=str, default='./data',
                      help='Dataset root directory')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=None,
                      help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=None,
                      help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                      help='Initial learning rate')
    
    # Model
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint')
    parser.add_argument('--pretrained', type=str, default=None,
                      help='Load pretrained weights')
    
    # System
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                      help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='./logs',
                      help='Log directory')
    parser.add_argument('--num-workers', type=int, default=8,
                      help='Number of data loading workers')
    parser.add_argument('--print-freq', type=int, default=10,
                      help='Print frequency')
    
    args = parser.parse_args()
    return args


def train_epoch(model, train_loader, criterion, optimizer, scaler, 
                epoch, cfg, logger, writer):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    bce_losses = AverageMeter()
    dice_losses = AverageMeter()
    ious = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{cfg.TRAIN.EPOCHS}')
    
    for i, batch in enumerate(pbar):
        images = batch['image'].cuda(non_blocking=True)
        masks = batch['mask'].cuda(non_blocking=True)
        
        # Mixed precision training
        with autocast(enabled=False):  # Force disable AMP to prevent NaN
            outputs = model(images)
            # TEMPORARY FIX: Force resize output to match target size
            if outputs.shape[2:] != masks.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[2:],
                    mode="bilinear", align_corners=False
                )
            loss, loss_dict = criterion(outputs, masks)
        
        # Backward
        optimizer.zero_grad()
        if cfg.TRAIN.USE_AMP:
            scaler.scale(loss).backward()
            if cfg.TRAIN.CLIP_GRAD_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                             cfg.TRAIN.CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.TRAIN.CLIP_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                             cfg.TRAIN.CLIP_GRAD_NORM)
            optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_binary = (outputs > 0.5).float()
            iou = calculate_metrics(pred_binary, masks)['iou']
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        bce_losses.update(loss_dict['bce'], images.size(0))
        dice_losses.update(loss_dict['dice'], images.size(0))
        ious.update(iou, images.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'iou': f'{ious.avg:.4f}'
        })
        
        # Logging
        if i % cfg.LOG.PRINT_FREQ == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch: [{epoch}/{cfg.TRAIN.EPOCHS}][{i}/{len(train_loader)}] '
                f'Loss: {losses.avg:.4f} | BCE: {bce_losses.avg:.4f} | '
                f'Dice: {dice_losses.avg:.4f} | IoU: {ious.avg:.4f} | '
                f'LR: {current_lr:.6f}'
            )
    
    # TensorBoard logging
    if writer is not None:
        global_step = epoch * len(train_loader)
        writer.add_scalar('train/loss', losses.avg, global_step)
        writer.add_scalar('train/bce_loss', bce_losses.avg, global_step)
        writer.add_scalar('train/dice_loss', dice_losses.avg, global_step)
        writer.add_scalar('train/iou', ious.avg, global_step)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    return {
        'loss': losses.avg,
        'iou': ious.avg
    }


def validate(model, val_loader, criterion, epoch, cfg, logger, writer):
    """验证"""
    model.eval()
    
    losses = AverageMeter()
    ious = AverageMeter()
    precisions = AverageMeter()
    recalls = AverageMeter()
    f1s = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating')
        for batch in pbar:
            images = batch['image'].cuda(non_blocking=True)
            masks = batch['mask'].cuda(non_blocking=True)
            
            # Forward
            outputs = model(images)
            # TEMPORARY FIX: Force resize output to match target size
            if outputs.shape[2:] != masks.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=masks.shape[2:],
                    mode="bilinear", align_corners=False
                )
            loss, _ = criterion(outputs, masks)
            
            # Metrics
            pred_binary = (outputs > 0.5).float()
            metrics = calculate_metrics(pred_binary, masks)
            
            # Update meters
            losses.update(loss.item(), images.size(0))
            ious.update(metrics['iou'], images.size(0))
            precisions.update(metrics['precision'], images.size(0))
            recalls.update(metrics['recall'], images.size(0))
            f1s.update(metrics['f1'], images.size(0))
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'iou': f'{ious.avg:.4f}'
            })
    
    # Logging
    logger.info(
        f'Validation: Epoch: [{epoch}/{cfg.TRAIN.EPOCHS}] '
        f'Loss: {losses.avg:.4f} | IoU: {ious.avg:.4f} | '
        f'Precision: {precisions.avg:.4f} | Recall: {recalls.avg:.4f} | '
        f'F1: {f1s.avg:.4f}'
    )
    
    # TensorBoard logging
    if writer is not None:
        writer.add_scalar('val/loss', losses.avg, epoch)
        writer.add_scalar('val/iou', ious.avg, epoch)
        writer.add_scalar('val/precision', precisions.avg, epoch)
        writer.add_scalar('val/recall', recalls.avg, epoch)
        writer.add_scalar('val/f1', f1s.avg, epoch)
    
    return {
        'loss': losses.avg,
        'iou': ious.avg,
        'precision': precisions.avg,
        'recall': recalls.avg,
        'f1': f1s.avg
    }


def main():
    # Parse arguments
    args = parse_args()
    
    # Update config
    update_config(cfg, args)
    
    # Set random seed
    set_seed(cfg.SYSTEM.SEED)
    
    # Create directories
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR, exist_ok=True)
    
    # Create logger
    logger = create_logger(cfg.LOG_DIR, 'train')
    logger.info("=" * 80)
    logger.info("SKD-SegFormer Training")
    logger.info("=" * 80)
    logger.info(f"Config: {cfg}")
    
    # Create TensorBoard writer
    writer = None
    if cfg.LOG.TENSORBOARD:
        writer = SummaryWriter(log_dir=os.path.join(cfg.LOG_DIR, 'tensorboard'))
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(cfg)
    
    # Create model
    logger.info("Creating model...")
    model = SKDSegFormer(cfg).cuda()
    logger.info(f"Model parameters: {model.get_param_count() / 1e6:.2f}M")
    
    # Create loss function
    criterion = get_loss_function(cfg).cuda()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        betas=cfg.TRAIN.BETAS
    )
    
    # Create learning rate scheduler
    scheduler = PolyLRScheduler(
        optimizer,
        max_epochs=cfg.TRAIN.EPOCHS,
        warmup_epochs=cfg.TRAIN.WARMUP_EPOCHS,
        warmup_lr=cfg.TRAIN.WARMUP_LR,
        min_lr=cfg.TRAIN.MIN_LR,
        power=cfg.TRAIN.POLY_POWER
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=cfg.TRAIN.USE_AMP)
    
    # Resume from checkpoint
    start_epoch = cfg.TRAIN.START_EPOCH
    best_iou = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_iou = checkpoint.get('best_iou', 0.0)
    
    elif args.pretrained:
        logger.info(f"Loading pretrained weights: {args.pretrained}")
        checkpoint = load_checkpoint(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, cfg, logger, writer
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        if (epoch + 1) % cfg.VAL.FREQUENCY == 0:
            val_metrics = validate(
                model, val_loader, criterion, epoch, cfg, logger, writer
            )
            
            # Save checkpoint
            is_best = val_metrics['iou'] > best_iou
            if is_best:
                best_iou = val_metrics['iou']
            
            if (epoch + 1) % cfg.LOG.SAVE_FREQ == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_iou': best_iou,
                    'config': cfg
                }, is_best, cfg.CHECKPOINT_DIR)
                
                logger.info(f"Checkpoint saved at epoch {epoch + 1}")
    
    logger.info("=" * 80)
    logger.info(f"Training completed! Best IoU: {best_iou:.4f}")
    logger.info("=" * 80)
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
