"""
SKD-SegFormer Configuration File
配置所有训练和测试参数
"""
import os
from easydict import EasyDict as edict

cfg = edict()

# ==================== Paths ====================
cfg.DATA_ROOT = './data'
cfg.TRAIN_DIR = os.path.join(cfg.DATA_ROOT, 'train')
cfg.VAL_DIR = os.path.join(cfg.DATA_ROOT, 'val')
cfg.TEST_DIR = os.path.join(cfg.DATA_ROOT, 'test')

cfg.CHECKPOINT_DIR = './checkpoints'
cfg.LOG_DIR = './logs'
cfg.RESULT_DIR = './results'

# ==================== Dataset ====================
cfg.DATASET = edict()
cfg.DATASET.NAME = 'SmokeSegmentation'
cfg.DATASET.IMG_SIZE = (512, 512)  # (H, W)
cfg.DATASET.NUM_CLASSES = 1  # Binary segmentation
cfg.DATASET.MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
cfg.DATASET.STD = [0.229, 0.224, 0.225]   # ImageNet std

# ==================== Model ====================
cfg.MODEL = edict()
cfg.MODEL.NAME = 'SKD-SegFormer'

# Encoder (SegFormer-B0)
cfg.MODEL.ENCODER = edict()
cfg.MODEL.ENCODER.EMBED_DIMS = [32, 64, 160, 256]  # Channel dimensions
cfg.MODEL.ENCODER.NUM_HEADS = [1, 2, 5, 8]         # Attention heads
cfg.MODEL.ENCODER.MLP_RATIOS = [4, 4, 4, 4]        # MLP expansion ratios
cfg.MODEL.ENCODER.QKV_BIAS = True
cfg.MODEL.ENCODER.DEPTHS = [2, 2, 2, 2]            # Number of blocks per stage
cfg.MODEL.ENCODER.SR_RATIOS = [8, 4, 2, 1]         # Spatial reduction ratios
cfg.MODEL.ENCODER.DROP_RATE = 0.0
cfg.MODEL.ENCODER.ATTN_DROP_RATE = 0.0
cfg.MODEL.ENCODER.DROP_PATH_RATE = 0.1

# KAN-MLP Head
cfg.MODEL.KAN = edict()
cfg.MODEL.KAN.IN_CHANNELS = [32, 64, 160, 256]
cfg.MODEL.KAN.OUT_CHANNELS = 256
cfg.MODEL.KAN.NUM_BASIS = 8  # G = 8 B-spline basis functions
cfg.MODEL.KAN.SPLINE_ORDER = 3  # Cubic B-splines

# DSDM (Dual-Spectrum Discrimination Module)
cfg.MODEL.DSDM = edict()
cfg.MODEL.DSDM.IN_CHANNELS = 256
cfg.MODEL.DSDM.DILATION_RATE_WHITE = 2  # For white smoke
cfg.MODEL.DSDM.DILATION_RATE_BLACK = 1  # For black smoke
cfg.MODEL.DSDM.CA_REDUCTION = 4  # Channel attention reduction ratio (256->64->256)

# SOSN (Smoke-Oriented Suppression Network)
cfg.MODEL.SOSN = edict()
cfg.MODEL.SOSN.IN_CHANNELS = 256
cfg.MODEL.SOSN.HIDDEN_CHANNELS = [128, 64, 1]
cfg.MODEL.SOSN.THRESHOLD_INIT = 0.5  # Initial threshold τ
cfg.MODEL.SOSN.USE_POS_ENCODING = True

# Decoder
cfg.MODEL.DECODER = edict()
cfg.MODEL.DECODER.IN_CHANNELS = 256
cfg.MODEL.DECODER.CHANNELS = [128, 64, 32, 16]

# ==================== Training ====================
cfg.TRAIN = edict()
cfg.TRAIN.BATCH_SIZE = 8  # Per GPU
cfg.TRAIN.NUM_WORKERS = 8
cfg.TRAIN.EPOCHS = 200
cfg.TRAIN.START_EPOCH = 0

# Optimizer
cfg.TRAIN.OPTIMIZER = 'AdamW'
cfg.TRAIN.LR = 6e-4  # Initial learning rate
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.BETAS = (0.9, 0.999)

# Learning Rate Scheduler
cfg.TRAIN.LR_SCHEDULER = 'poly'  # 'poly', 'cosine', 'step'
cfg.TRAIN.WARMUP_EPOCHS = 10
cfg.TRAIN.WARMUP_LR = 1e-6
cfg.TRAIN.MIN_LR = 1e-6
cfg.TRAIN.POLY_POWER = 0.9

# Loss Function
cfg.TRAIN.LOSS = edict()
cfg.TRAIN.LOSS.BCE_WEIGHT = 0.5  # λ in paper
cfg.TRAIN.LOSS.DICE_WEIGHT = 0.5  # 1-λ in paper
cfg.TRAIN.LOSS.BOUNDARY_WEIGHT = 3.0  # w_b for boundary pixels
cfg.TRAIN.LOSS.KERNEL_SIZE = 3  # Morphological operation kernel

# Augmentation
cfg.TRAIN.AUGMENTATION = edict()
cfg.TRAIN.AUGMENTATION.RANDOM_FLIP = True
cfg.TRAIN.AUGMENTATION.RANDOM_ROTATE = True
cfg.TRAIN.AUGMENTATION.ROTATE_LIMIT = 30
cfg.TRAIN.AUGMENTATION.COLOR_JITTER = True
cfg.TRAIN.AUGMENTATION.BRIGHTNESS = 0.2
cfg.TRAIN.AUGMENTATION.CONTRAST = 0.2
cfg.TRAIN.AUGMENTATION.SATURATION = 0.2

# Gradient Clipping
cfg.TRAIN.CLIP_GRAD_NORM = 1.0

# Mixed Precision Training
cfg.TRAIN.USE_AMP = True  # Automatic Mixed Precision

# ==================== Validation ====================
cfg.VAL = edict()
cfg.VAL.BATCH_SIZE = 8
cfg.VAL.NUM_WORKERS = 4
cfg.VAL.FREQUENCY = 1  # Validate every N epochs

# ==================== Testing ====================
cfg.TEST = edict()
cfg.TEST.BATCH_SIZE = 1
cfg.TEST.NUM_WORKERS = 4
cfg.TEST.SAVE_RESULTS = True
cfg.TEST.SAVE_OVERLAY = True  # Save prediction overlay on original image
cfg.TEST.CHECKPOINT = 'best_model.pth'

# ==================== System ====================
cfg.SYSTEM = edict()
cfg.SYSTEM.NUM_GPUS = 1
cfg.SYSTEM.DEVICE = 'cuda'
cfg.SYSTEM.SEED = 42
cfg.SYSTEM.CUDNN_BENCHMARK = True
cfg.SYSTEM.CUDNN_DETERMINISTIC = False

# ==================== Logging ====================
cfg.LOG = edict()
cfg.LOG.PRINT_FREQ = 10  # Print every N iterations
cfg.LOG.SAVE_FREQ = 10   # Save checkpoint every N epochs
cfg.LOG.TENSORBOARD = True

# ==================== Distributed Training (Optional) ====================
cfg.DDP = edict()
cfg.DDP.ENABLE = False  # Set to True for multi-GPU training
cfg.DDP.LOCAL_RANK = 0
cfg.DDP.WORLD_SIZE = 1

def update_config(cfg, args):
    """Update config from command line arguments"""
    if args.batch_size:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs:
        cfg.TRAIN.EPOCHS = args.epochs
    if args.lr:
        cfg.TRAIN.LR = args.lr
    if args.checkpoint_dir:
        cfg.CHECKPOINT_DIR = args.checkpoint_dir
    if args.data_root:
        cfg.DATA_ROOT = args.data_root
        cfg.TRAIN_DIR = os.path.join(cfg.DATA_ROOT, 'train')
        cfg.VAL_DIR = os.path.join(cfg.DATA_ROOT, 'val')
        cfg.TEST_DIR = os.path.join(cfg.DATA_ROOT, 'test')
    
    return cfg

if __name__ == '__main__':
    print("=" * 80)
    print("SKD-SegFormer Configuration")
    print("=" * 80)
    
    print("\n[Model Configuration]")
    print(f"  Encoder Channels: {cfg.MODEL.ENCODER.EMBED_DIMS}")
    print(f"  KAN Basis Functions: {cfg.MODEL.KAN.NUM_BASIS}")
    print(f"  DSDM Dilation Rates: White={cfg.MODEL.DSDM.DILATION_RATE_WHITE}, Black={cfg.MODEL.DSDM.DILATION_RATE_BLACK}")
    print(f"  SOSN Threshold: {cfg.MODEL.SOSN.THRESHOLD_INIT}")
    
    print("\n[Training Configuration]")
    print(f"  Batch Size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"  Epochs: {cfg.TRAIN.EPOCHS}")
    print(f"  Learning Rate: {cfg.TRAIN.LR}")
    print(f"  Loss Weights: BCE={cfg.TRAIN.LOSS.BCE_WEIGHT}, Dice={cfg.TRAIN.LOSS.DICE_WEIGHT}")
    print(f"  Boundary Weight: {cfg.TRAIN.LOSS.BOUNDARY_WEIGHT}")
    
    print("\n[System Configuration]")
    print(f"  GPU: NVIDIA Tesla A40 (48GB)")
    print(f"  CUDA: 11.8")
    print(f"  PyTorch: 2.0.1")
    print(f"  Mixed Precision: {cfg.TRAIN.USE_AMP}")
    
    print("=" * 80)
