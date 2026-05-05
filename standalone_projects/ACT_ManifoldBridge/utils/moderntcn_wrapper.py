import torch
import torch.nn as nn
from types import SimpleNamespace
from core.moderntcn import Model as ModernTCNModel

class ModernTCNClassifier(nn.Module):
    def __init__(self, c_in, seq_len, num_classes):
        super().__init__()
        
        # Dynamically adjust architecture based on sequence length to avoid shape errors
        patch_size = 8
        patch_stride = 4
        
        if seq_len < 32:
            patch_size = 2
            patch_stride = 1
            num_blocks = [1, 1]
            large_size = [13, 11]
            small_size = [5, 5]
            dims = [64, 128]
            dw_dims = [64, 128]
        elif seq_len < 96:
            patch_size = 4
            patch_stride = 2
            num_blocks = [1, 1, 1]
            large_size = [17, 15, 13]
            small_size = [5, 5, 5]
            dims = [64, 64, 128]
            dw_dims = [64, 64, 128]
        else:
            num_blocks = [1, 1, 1, 1]
            large_size = [31, 29, 27, 13]
            small_size = [5, 5, 5, 5]
            dims = [64, 64, 64, 64]
            dw_dims = [64, 64, 64, 64]

        configs = SimpleNamespace(
            task_name='classification',
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=2,
            num_blocks=num_blocks,
            large_size=large_size,
            small_size=small_size,
            dims=dims,
            dw_dims=dw_dims,
            enc_in=c_in,
            small_kernel_merged=False,
            dropout=0.1,
            head_dropout=0.0,
            use_multi_scale=False,
            revin=True,
            affine=True,
            subtract_last=False,
            freq='h',
            seq_len=seq_len,
            individual=False,
            pred_len=96,
            kernel_size=5,
            patch_size=patch_size,
            patch_stride=patch_stride,
            class_dropout=0.1,
            num_class=num_classes,
            decomposition=False
        )
        self.model = ModernTCNModel(configs)
        
    def forward(self, x):
        # x is [B, C, L]
        # ModernTCN expects [B, L, C] inside forward (it will permute internally if needed, 
        # wait! Let's check: core/moderntcn.py line 508 does x = x.permute(0, 2, 1) assuming input is [B, L, C]!)
        # If our framework passes [B, C, L], and ModernTCN permutes it to [B, C, L] internally assuming it was [B, L, C]...
        # Wait! If input is [B, C, L], we should permute it to [B, L, C] so ModernTCN permutes it BACK to [B, C, L].
        # Let's permute just in case.
        x = x.permute(0, 2, 1) # to [B, L, C]
        return self.model(x, None, None, None)
