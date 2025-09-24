import MinkowskiEngine as ME
import torch.nn as nn
import torch
from .minkunet import MinkUNetBase,ResNetBase
from MinkowskiEngine.modules.resnet_block import BasicBlock

class MinkUNet34Encoder(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)

    def forward(self, x, is_seg=False):
        # 输出 feature dimension 256
        if not is_seg:
            _, out = super().forward(x, is_seg)
            # output feature dimension 256
            return out
        else:
            out = super().forward(x,is_seg)
            return out
