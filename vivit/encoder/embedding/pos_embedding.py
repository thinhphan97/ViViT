import torch
import torch.nn as nn
import torch.nn.functional as F


from ...utils.utils import pair, trunc_normal_

class PosEmbedding(nn.Module):
    def __init__(self, shape, dim, drop=None, sinusoidal=False, std=0.02):
        super(PosEmbedding, self).__init__()

        if not sinusoidal:
            if isinstance(shape, int):
                shape = [1, shape, dim]
            else:
                shape = [1] + list(shape) + [dim]
            self.pos_embed = nn.Parameter(torch.zeros(shape))

        else:
            pe = torch.FloatTensor(
                [
                    [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                    for p in range(shape)
                ]
            )
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            self.pos_embed = pe
            self.pos_embed.requires_grad = False
        trunc_normal_(self.pos_embed, std=std)
        self.pos_drop = nn.Dropout(drop) if drop is not None else nn.Identity()

    def forward(self, x):
        x = x + self.pos_embed
        return self.pos_drop(x)