import torch
import torch.nn as nn
from einops import rearrange, repeat

from ..utils import MODEL_REGISTRY, trunc_normal_
from ..common import *
from ..encoder import *
from ..decoder import *


@MODEL_REGISTRY.register()
class CrossStageTransformer(BaseClassificationModel):
    """
    Model cross-stage transformation implementation of Cross-Stage Transformer for Video Learning
    Parameters
    ----------
    img_size:int
        Size of single frame/ image in video
    in_channels:int
        Number of channels
    patch_t:int
        Temporal length of single tube/patch in tubelet embedding
    patch_h:int
        Height  of single tube/patch in tubelet embedding
    patch_w:int
        Width  of single tube/patch in tubelet embedding
    tubelet: bool
        True using Tubelet in patch embedding
    embedding_dim: int
        Embedding dimension of a patch
    num_frames:int
        Number of seconds in each Video
    depth:int
        Number of encoder layers
    num_heads:int
        Number of attention heads
    head_dim:int
        Dimension of head
    n_classes:int
        Number of classes
    mlp_dim: int
        Dimension of hidden layer
    pool: str
        Pooling operation,must be one of {"cls","mean"},default is "cls"
    p_dropout:float
        Dropout probability
    attn_dropout:float
        Dropout probability
    drop_path_rate:float
        Stochastic drop path rate
    """
    def __init__(self,
                img_size,
                in_channels,
                patch_t,
                patch_h,
                patch_w,
                tubelet,
                embedding_dim,
                num_frames,
                depth_STB,
                depth_TTB,
                num_heads,
                head_dim,
                n_classes,
                mlp_dim_STB=None,
                mlp_dim_TTB=None,
                pool="cls",
                p_dropout=0.0):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            patch_size= (patch_h, patch_w),
            pool=pool,)
        h, w = pair(img_size)
        
        if tubelet:
            assert num_frames % patch_t == 0, 'Time dimesions must be divisible by the patch time.'
            self.patch_embedding = TubeletEmbedding(embedding_dim=embedding_dim, 
                                                    tubelet_t=patch_t, 
                                                    tubelet_h=self.patch_height, 
                                                    tubelet_w=self.patch_width, 
                                                    in_channels=in_channels)
            self.pos_embedding = PosEmbedding(shape=[num_frames//patch_t, (h * w) // (patch_w * patch_h)], dim=embedding_dim, drop=p_dropout)
        else:
            patch_dim = in_channels*self.patch_height*self.patch_width
            self.patch_embedding =  LinearVideoEmbedding( embedding_dim=embedding_dim, 
                                                         patch_height=self.patch_height, 
                                                         patch_width=self.patch_width, 
                                                         patch_dim=patch_dim,)
            self.pos_embedding = PosEmbedding(shape=[num_frames, (h * w) // (patch_w * patch_h)], dim=embedding_dim, drop=p_dropout)
        self.spatial_transformer = CrosStageTransformerHalf(
            dim=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            p_dropout=p_dropout,
            depth=depth_STB,
            hidden_dim=mlp_dim_STB,
            STB = True,
        )

        self.time_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.temporal_transformer = CrosStageTransformerHalf(
            dim=embedding_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            p_dropout=p_dropout,
            depth=depth_TTB,
            hidden_dim=mlp_dim_TTB,
            STB = False,
        )

        self.decoder = MLPDecoder(
            config=[
                embedding_dim,
            ],
            n_classes=n_classes,
        )
        self.beta = trunc_normal_(torch.nn.Parameter(torch.zeros((depth_TTB + depth_STB,1,1,1,1))), std= 0.02)
        self.norm = nn.LayerNorm(embedding_dim)
    def forward(self, x):
        """ x is a video: (B, T, C, H, W) """
        
        b = x.shape[0]
        x = self.patch_embedding(x)
    
        x = self.pos_embedding(x)
        x, y = self.spatial_transformer(x) 
        _, yt = self.temporal_transformer(x)
        
        y.extend(yt)
        y = torch.stack(y,dim = 0)
        
        y = self.norm(self.beta*y)
        
        y = y.sum(dim=0)
        y = y.reshape(b, -1, y.shape[-1])
        y = y.mean(dim=1) if self.pool == 'mean' else x[:,0, :]
        y = self.decoder(y)      
        return y
        