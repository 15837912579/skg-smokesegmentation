"""
数据集预处理脚本
用于转换和验证数据集格式
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path


def convert_mask_to_binary(mask_path, output_path, threshold=127):
    """
    将掩码转换为二值图像 (0和255)
    
    Args:
        mask_path: 输入掩码路径
        output_path: 输出掩码路径
        threshold: 二值化阈值
    """
    # 读取掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"警告: 无法读取 {mask_path}")
        return False
    
    # 二值化
    _, binary_mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    
    # 保存
    cv2.imwrite(output_path, binary_mask)
    return True


def check_dataset(data_root):
    """检查数据集的完整性"""
    print("\n" + "="*60)
    print("检查数据集")
    print("="*60)
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        img_dir = os.path.join(data_root, split, 'images')
        mask_dir = os.path.join(data_root, split, 'masks')
        
        if not os.path.exists(img_dir):
            print(f"⚠️  {split}/images 目录不存在")
            continue
        
        if not os.path.exists(mask_dir):
            print(f"⚠️  {split}/masks 目录不存在")
            continue
        
        # 统计文件数量
        img_files = set([f for f in os.listdir(img_dir) 
                        if f.endswith(('.jpg', '.png', '.jpeg'))])
        mask_files = set([f for f in os.listdir(mask_dir) 
                         if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        # 检查文件名匹配
        img_names = set([os.path.splitext(f)[0] for f in img_files])
        mask_names = set([os.path.splitext(f)[0] for f in mask_files])
        
        matched = img_names & mask_names
        only_img = img_names - mask_names
        only_mask = mask_names - img_names
        
        stats[split] = {
            'images': len(img_files),
            'masks': len(mask_files),
            'matched': len(matched),
            'only_img': len(only_img),
            'only_mask': len(only_mask)
        }
        
        print(f"\n[{split.upper()}]")
        print(f"  图像数量: {len(img_files)}")
        print(f"  掩码数量: {len(mask_files)}")
        print(f"  ✓ 匹配: {len(matched)}")
        
        if only_img:
            print(f"  ⚠️  仅有图像: {len(only_img)}")
        if only_mask:
            print(f"  ⚠️  仅有掩码: {len(only_mask)}")
    
    print("\n" + "="*60)
    return stats


def validate_masks(data_root, split='train'):
    """验证掩码的像素值"""
    print(f"\n验证 {split} 掩码...")
    
    mask_dir = os.path.join(data_root, split, 'masks')
    if not os.path.exists(mask_dir):
        print(f"目录不存在: {mask_dir}")
        return
    
    mask_files = [f for f in os.listdir(mask_dir) 
                 if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    invalid_masks = []
    
    for mask_file in tqdm(mask_files[:100], desc="验证掩码"):  # 只检查前100个
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
        
        unique_values = np.unique(mask)
        
        # 检查是否只有0和255
        if not np.all((unique_values == 0) | (unique_values == 255)):
            invalid_masks.append({
                'file': mask_file,
                'unique_values': unique_values.tolist(),
                'min': mask.min(),
                'max': mask.max()
            })
    
    if invalid_masks:
        print(f"\n⚠️  发现 {len(invalid_masks)} 个非二值掩码:")
        for i, info in enumerate(invalid_masks[:5]):  # 只显示前5个
            print(f"  {i+1}. {info['file']}")
            print(f"     唯一值: {info['unique_values']}")
            print(f"     范围: [{info['min']}, {info['max']}]")
        
        if len(invalid_masks) > 5:
            print(f"  ... 还有 {len(invalid_masks) - 5} 个")
        
        return False
    else:
        print("✓ 所有掩码都是二值图像 (0和255)")
        return True


def convert_all_masks(data_root, splits=['train', 'val', 'test'], threshold=127):
    """转换所有掩码为二值图像"""
    print("\n" + "="*60)
    print("转换掩码为二值图像")
    print("="*60)
    
    for split in splits:
        mask_dir = os.path.join(data_root, split, 'masks')
        
        if not os.path.exists(mask_dir):
            print(f"⚠️  跳过 {split}: 目录不存在")
            continue
        
        # 创建备份目录
        backup_dir = os.path.join(data_root, split, 'masks_backup')
        os.makedirs(backup_dir, exist_ok=True)
        
        mask_files = [f for f in os.listdir(mask_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n[{split.upper()}] 处理 {len(mask_files)} 个掩码...")
        
        success_count = 0
        for mask_file in tqdm(mask_files):
            mask_path = os.path.join(mask_dir, mask_file)
            backup_path = os.path.join(backup_dir, mask_file)
            
            # 备份原始文件
            os.system(f'cp "{mask_path}" "{backup_path}"')
            
            # 转换
            if convert_mask_to_binary(mask_path, mask_path, threshold):
                success_count += 1
        
        print(f"✓ 成功转换 {success_count}/{len(mask_files)} 个掩码")
        print(f"  原始文件备份至: {backup_dir}")


def visualize_samples(data_root, split='train', num_samples=5):
    """可视化样本"""
    print(f"\n可视化 {split} 样本...")
    
    img_dir = os.path.join(data_root, split, 'images')
    mask_dir = os.path.join(data_root, split, 'masks')
    output_dir = os.path.join(data_root, f'{split}_visualization')
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_files = sorted([f for f in os.listdir(img_dir) 
                       if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for i, img_file in enumerate(img_files[:num_samples]):
        # 读取图像
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        
        # 读取掩码
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(mask_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # 尝试其他扩展名
            for ext in ['.jpg', '.jpeg']:
                alt_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + ext)
                if os.path.exists(alt_path):
                    mask_path = alt_path
                    break
        
        if not os.path.exists(mask_path):
            print(f"⚠️  找不到掩码: {mask_name}")
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 创建可视化
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # 叠加
        overlay = img.copy()
        overlay[mask > 127] = overlay[mask > 127] * 0.5 + np.array([0, 255, 0]) * 0.5
        
        # 拼接
        vis = np.hstack([img, mask_rgb, overlay])
        
        # 保存
        output_path = os.path.join(output_dir, f'sample_{i+1}.jpg')
        cv2.imwrite(output_path, vis)
    
    print(f"✓ 可视化保存至: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='数据集预处理工具')
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据集根目录')
    parser.add_argument('--check', action='store_true',
                       help='检查数据集')
    parser.add_argument('--validate', action='store_true',
                       help='验证掩码格式')
    parser.add_argument('--convert', action='store_true',
                       help='转换掩码为二值图像')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化样本')
    parser.add_argument('--threshold', type=int, default=127,
                       help='二值化阈值 (默认127)')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='可视化样本数量')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SKD-SegFormer 数据集预处理工具")
    print("="*60)
    print(f"数据集路径: {args.data_root}")
    print()
    
    if not os.path.exists(args.data_root):
        print(f"错误: 数据集目录不存在: {args.data_root}")
        return
    
    # 如果没有指定任何操作，默认执行检查
    if not (args.check or args.validate or args.convert or args.visualize):
        args.check = True
        args.validate = True
    
    # 检查数据集
    if args.check:
        check_dataset(args.data_root)
    
    # 验证掩码
    if args.validate:
        for split in ['train', 'val', 'test']:
            validate_masks(args.data_root, split)
    
    # 转换掩码
    if args.convert:
        convert_all_masks(args.data_root, threshold=args.threshold)
    
    # 可视化
    if args.visualize:
        for split in ['train', 'val', 'test']:
            visualize_samples(args.data_root, split, args.num_samples)
    
    print("\n" + "="*60)
    print("处理完成!")
    print("="*60)


if __name__ == '__main__':
    main()
