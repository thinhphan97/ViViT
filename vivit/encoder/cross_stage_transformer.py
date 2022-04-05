from re import X
import torch
import torch.nn as nn
from ..utils import ENCODER_REGISTRY
from ..attention import CrossStageSelfAttention
from .nn import FeedForward


@ENCODER_REGISTRY.register()
class CrosStageTransformerBlock(nn.Module):
    """for model cross-stage transformer"""

    def __init__(self, dim, num_heads, head_dim, p_dropout, STB=True, out_dim=None, hidden_dim=None):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.msa = CrossStageSelfAttention(dim, num_heads, head_dim, p_dropout)
        
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.STB = STB

    def forward(self, x, A):
        
        b, n, s, d = x.shape
        x = self.norm(x)

        if not self.STB:
            x = x.reshape(b, n, s, d).transpose(1, 2)

        x = torch.flatten(x, start_dim=0, end_dim=1)

        x, A = self.msa(x, A)
        x += x

        x = self.norm(x)
        x = self.mlp(x) + x

        x = x.reshape(
            b, n, s, d
        )  # reshaping because this block is used for several depths in ViViTEncoder class and Next layer will expect the x in proper shape

        return x, A
    
@ENCODER_REGISTRY.register()
class CrosStageTransformerHalf(nn.Module):
    """for model cross-stage transformer"""
    def __init__(self, dim, num_heads, head_dim, p_dropout, depth, STB=True, out_dim=None, hidden_dim=None):
        super().__init__()
        
        self.encoder = nn.ModuleList()

        for _ in range(depth):
            self.encoder.append(
                CrosStageTransformerBlock(
                    dim, num_heads, head_dim, p_dropout, STB, out_dim, hidden_dim
                )
            )
    
    def forward(self, x):
        b = x.shape[0]
        Y = []
        A = None
        
        for blk in self.encoder:
            
            x, A = blk(x, A)
            Y.append(x)
            
        return x, Y