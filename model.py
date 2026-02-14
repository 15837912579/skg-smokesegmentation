"""
SKD-SegFormer Complete Model Implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ==================== Encoder: SegFormer-B0 ====================

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                            kernel_size=patch_size, 
                            stride=stride,
                            padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        _, _, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
            x_ = self.sr(x_)
            x_ = rearrange(x_, 'b c h w -> b (h w) c')
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MixFFN(nn.Module):
    """Mix-FFN: Position-aware Feature Transformation"""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, 
                               groups=hidden_features, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x, H, W):
        x = self.fc1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.dwconv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads, qkv_bias, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio))
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)
        return x


class SegFormerEncoder(nn.Module):
    """SegFormer-B0 Encoder"""
    def __init__(self, in_chans=3, embed_dims=[32, 64, 160, 256], 
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True, depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        
        # Patch embeddings for each stage
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        
        # Transformer blocks for each stage
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, sr_ratios[3])
            for _ in range(depths[3])
        ])
        
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])
    
    def forward(self, x):
        B = x.shape[0]
        outs = []
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        outs.append(x)
        
        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        outs.append(x)
        
        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        outs.append(x)
        
        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        outs.append(x)
        
        return outs  # [F1, F2, F3, F4]


# ==================== KAN-MLP Head ====================

class BSplineKANLayer(nn.Module):
    """B-spline KAN Layer for channel-wise adaptive activation"""
    def __init__(self, num_channels, num_basis=8, spline_order=3):
        super().__init__()
        self.num_channels = num_channels
        self.num_basis = num_basis
        self.spline_order = spline_order
        
        # Learnable coefficients for each channel
        self.coeffs = nn.Parameter(torch.randn(num_channels, num_basis))
        
        # B-spline basis functions (uniformly distributed)
        grid = torch.linspace(-1, 1, num_basis + spline_order + 1)
        self.register_buffer('grid', grid)
        
    def forward(self, x):
        """
        x: (B, C, H, W) or (B, N, C)
        """
        original_shape = x.shape
        if len(x.shape) == 4:  # (B, C, H, W)
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> (b h w) c')
        else:  # (B, N, C)
            B, N, C = x.shape
            x = rearrange(x, 'b n c -> (b n) c')
        
        N_points, C = x.shape
        
        # Normalize input to [-1, 1]
        x_norm = torch.tanh(x)  # (N, C)
        
        # Compute B-spline basis activation
        # Map x_norm to [0, num_basis-1] range
        x_scaled = (x_norm + 1) / 2 * (self.num_basis - 1)  # (N, C)
        
        # Initialize output
        kan_output = torch.zeros_like(x)  # (N, C)
        
        # For each basis function
        for i in range(self.num_basis):
            # Distance from basis center
            t = x_scaled - i  # (N, C)
            t_abs = t.abs()
            
            # Cubic B-spline kernel
            basis_val = torch.zeros_like(t)
            
            # |t| < 1: central region
            mask1 = t_abs < 1
            basis_val = torch.where(
                mask1,
                (2.0/3.0) - t_abs**2 + 0.5 * t_abs**3,
                basis_val
            )
            
            # 1 <= |t| < 2: outer region
            mask2 = (t_abs >= 1) & (t_abs < 2)
            basis_val = torch.where(
                mask2,
                (2.0 - t_abs)**3 / 6.0,
                basis_val
            )
            
            # Apply coefficients: basis_val (N, C) * coeffs[:, i] (C,)
            kan_output += basis_val * self.coeffs[:, i].unsqueeze(0)
        
        # Add residual connection
        output = x + kan_output
        
        # Restore original shape
        if len(original_shape) == 4:
            output = rearrange(output, '(b h w) c -> b c h w', b=B, h=H, w=W)
        else:
            output = rearrange(output, '(b n) c -> b n c', b=B, n=N)
        
        return output


class KANMLPHead(nn.Module):
    """KAN-Enhanced MLP Head"""
    def __init__(self, in_channels=[32, 64, 160, 256], out_channels=256, 
                 num_basis=8, spline_order=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Feature alignment and fusion
        self.align_ops = nn.ModuleList()
        for i, c in enumerate(in_channels):
            if i == 0:  # Downsample
                self.align_ops.append(nn.AvgPool2d(2, 2))
            elif i == 1:  # Keep
                self.align_ops.append(nn.Identity())
            else:  # Upsample
                self.align_ops.append(nn.Upsample(scale_factor=2**(i-1), mode='bilinear', align_corners=False))
        
        # 1x1 convolution for channel fusion
        total_channels = sum(in_channels)
        self.fusion_conv = nn.Conv2d(total_channels, out_channels, 1)
        
        # KAN layer for adaptive activation
        self.kan = BSplineKANLayer(out_channels, num_basis, spline_order)
        
        # MLP
        self.norm = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.GELU(),
            nn.Linear(out_channels * 4, out_channels)
        )
    
    def forward(self, features):
        """
        features: [F1, F2, F3, F4]
        F1: (B, 32, H/2, W/2)
        F2: (B, 64, H/4, W/4)
        F3: (B, 160, H/8, W/8)
        F4: (B, 256, H/16, W/16)
        """
        # Align all features to H/4 x W/4
        aligned = []
        for i, (feat, align_op) in enumerate(zip(features, self.align_ops)):
            aligned.append(align_op(feat))
        
        # Concatenate and fuse
        x = torch.cat(aligned, dim=1)  # (B, 512, H/4, W/4)
        x = self.fusion_conv(x)  # (B, 256, H/4, W/4)
        
        # Apply KAN activation
        x = self.kan(x)  # (B, 256, H/4, W/4)
        
        # MLP
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.mlp(self.norm(x))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x  # F_mlp


# ==================== DSDM: Dual-Spectrum Discrimination Module ====================

class ChannelAttention(nn.Module):
    """Channel Attention via Squeeze-and-Excitation"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SmokeBranch(nn.Module):
    """Single branch for white or black smoke"""
    def __init__(self, channels=256, dilation_rate=1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 
                             padding=dilation_rate, dilation=dilation_rate)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(channels, reduction=4)
    
    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.ca(out)
        out = out + identity  # Residual
        return out


class DSDM(nn.Module):
    """Dual-Spectrum Smoke Discrimination Module"""
    def __init__(self, channels=256, dilation_white=2, dilation_black=1):
        super().__init__()
        
        # Parallel branches
        self.white_branch = SmokeBranch(channels, dilation_white)
        self.black_branch = SmokeBranch(channels, dilation_black)
        
        # Adaptive gating network
        self.gate = nn.Sequential(
            nn.Conv2d(channels, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        x: F_mlp (B, 256, H/4, W/4)
        """
        # Parallel processing
        white_feat = self.white_branch(x)  # For white smoke
        black_feat = self.black_branch(x)  # For black smoke
        
        # Adaptive gating
        gate = self.gate(x)  # (B, 1, H/4, W/4)
        
        # Fusion
        out = gate * white_feat + (1 - gate) * black_feat
        
        return out  # F_dual


# ==================== SOSN: Smoke-Oriented Suppression Network ====================

class PositionEncoding2D(nn.Module):
    """2D Sinusoidal Position Encoding"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # Create frequency for half channels (will be used for both h and w)
        half_channels = channels // 2
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, half_channels, 2).float() / half_channels))
        self.register_buffer('inv_freq_buffer', self.inv_freq)
    
    def forward(self, x):
        B, C, H, W = x.shape
        pos_h = torch.arange(H, device=x.device).type_as(self.inv_freq_buffer)
        pos_w = torch.arange(W, device=x.device).type_as(self.inv_freq_buffer)
        
        # Compute sinusoidal encodings
        sin_inp_h = torch.einsum('i,j->ij', pos_h, self.inv_freq_buffer)
        sin_inp_w = torch.einsum('i,j->ij', pos_w, self.inv_freq_buffer)
        
        # Concatenate sin and cos for full encoding
        emb_h = torch.cat([sin_inp_h.sin(), sin_inp_h.cos()], dim=-1)  # (H, half_channels)
        emb_w = torch.cat([sin_inp_w.sin(), sin_inp_w.cos()], dim=-1)  # (W, half_channels)
        
        # Combine into 2D position encoding
        emb = torch.zeros((H, W, self.channels), device=x.device, dtype=x.dtype)
        emb[:, :, :self.channels//2] = emb_h.unsqueeze(1).repeat(1, W, 1)
        emb[:, :, self.channels//2:] = emb_w.unsqueeze(0).repeat(H, 1, 1)
        
        # Reshape to (B, C, H, W)
        return emb.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)


class SOSN(nn.Module):
    """Smoke-Oriented Suppression Network"""
    def __init__(self, channels=256, hidden_channels=[128, 64, 1], 
                 threshold_init=0.5):
        super().__init__()
        
        # Position encoding
        self.pos_encoding = PositionEncoding2D(channels)
        
        # Confidence discrimination head
        layers = []
        in_c = channels
        for out_c in hidden_channels:
            layers.extend([
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.BatchNorm2d(out_c) if out_c > 1 else nn.Identity(),
                nn.ReLU(inplace=True) if out_c > 1 else nn.Identity()
            ])
            in_c = out_c
        
        layers.append(nn.Sigmoid())
        self.confidence_head = nn.Sequential(*layers)
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
    
    def forward(self, x):
        """
        x: F_dual (B, 256, H/4, W/4)
        """
        # Add position encoding
        pos_enc = self.pos_encoding(x)
        x_pos = x + pos_enc
        
        # Confidence map
        confidence = self.confidence_head(x_pos)  # (B, 1, H/4, W/4)
        
        # Suppression mask
        mask = (confidence > self.threshold).float()
        
        # Apply suppression
        out = mask * x
        
        return out  # F_final


# ==================== Decoder ====================

class DecoderBlock(nn.Module):
    """Decoder block with skip connection"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class Decoder(nn.Module):
    """Lightweight Decoder"""
    def __init__(self, in_channels=256, encoder_channels=[32, 64, 160, 256],
                 decoder_channels=[128, 64, 32, 16]):
        super().__init__()
        
        # Features are fused in KAN-MLP at H/4 resolution
        # Need to upsample back to original resolution
        
        self.decoder1 = DecoderBlock(in_channels, 0, decoder_channels[0])  # 2x upsample
        self.decoder2 = DecoderBlock(decoder_channels[0], 0, decoder_channels[1])  # 2x upsample
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(decoder_channels[1], 1, 1)
    
    def forward(self, x, skip_connections=None):
        """
        x: F_final (B, 256, H/4, W/4)
        Returns: (B, 1, H, W) - Back to original input size
        """
        # Store original target size (will be 4x the input)
        target_h = x.shape[2] * 4
        target_w = x.shape[3] * 4
        
        x = self.decoder1(x, None)  # 2x upsample
        x = self.decoder2(x, None)  # 2x upsample (should be 4x total now)
        
        x = self.seg_head(x)  # (B, 1, H, W)
        
        # Ensure output matches target size (safety check)
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return x


# ==================== Complete SKD-SegFormer ====================

class SKDSegFormer(nn.Module):
    """Complete SKD-SegFormer Network"""
    def __init__(self, cfg):
        super().__init__()
        
        # Encoder
        self.encoder = SegFormerEncoder(
            in_chans=3,
            embed_dims=cfg.MODEL.ENCODER.EMBED_DIMS,
            num_heads=cfg.MODEL.ENCODER.NUM_HEADS,
            mlp_ratios=cfg.MODEL.ENCODER.MLP_RATIOS,
            qkv_bias=cfg.MODEL.ENCODER.QKV_BIAS,
            depths=cfg.MODEL.ENCODER.DEPTHS,
            sr_ratios=cfg.MODEL.ENCODER.SR_RATIOS
        )
        
        # KAN-MLP Head
        self.kan_mlp = KANMLPHead(
            in_channels=cfg.MODEL.KAN.IN_CHANNELS,
            out_channels=cfg.MODEL.KAN.OUT_CHANNELS,
            num_basis=cfg.MODEL.KAN.NUM_BASIS,
            spline_order=cfg.MODEL.KAN.SPLINE_ORDER
        )
        
        # DSDM
        self.dsdm = DSDM(
            channels=cfg.MODEL.DSDM.IN_CHANNELS,
            dilation_white=cfg.MODEL.DSDM.DILATION_RATE_WHITE,
            dilation_black=cfg.MODEL.DSDM.DILATION_RATE_BLACK
        )
        
        # SOSN
        self.sosn = SOSN(
            channels=cfg.MODEL.SOSN.IN_CHANNELS,
            hidden_channels=cfg.MODEL.SOSN.HIDDEN_CHANNELS,
            threshold_init=cfg.MODEL.SOSN.THRESHOLD_INIT
        )
        
        # Decoder
        self.decoder = Decoder(
            in_channels=cfg.MODEL.DECODER.IN_CHANNELS,
            encoder_channels=cfg.MODEL.ENCODER.EMBED_DIMS,
            decoder_channels=cfg.MODEL.DECODER.CHANNELS
        )
    
    def forward(self, x):
        """
        x: (B, 3, H, W)
        return: (B, 1, H, W) - Segmentation mask
        """
        # Encoder
        enc_features = self.encoder(x)  # [F1, F2, F3, F4]
        
        # KAN-MLP Head
        f_mlp = self.kan_mlp(enc_features)  # (B, 256, H/4, W/4)
        
        # DSDM
        f_dual = self.dsdm(f_mlp)  # (B, 256, H/4, W/4)
        
        # SOSN
        f_final = self.sosn(f_dual)  # (B, 256, H/4, W/4)
        
        # Decoder
        out = self.decoder(f_final, enc_features)  # (B, 1, H, W)
        
        return torch.sigmoid(out)  # Removed sigmoid (BCEWithLogitsLoss includes it)
    
    def get_param_count(self):
        """Calculate total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    """Test model"""
    import sys
    sys.path.append('.')
    from config import cfg
    
    print("=" * 60)
    print("Testing SKD-SegFormer Model")
    print("=" * 60)
    
    model = SKDSegFormer(cfg).cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    
    with torch.no_grad():
        out = model(x)
    
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"\nTotal parameters: {model.get_param_count() / 1e6:.2f}M")
    print("\nModel test completed successfully!")
    print("=" * 60)
