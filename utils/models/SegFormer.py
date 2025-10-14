from utils.modules.SegformerEncoder import mix_transformer
from utils.modules.SegformerDecoder import segformer_head
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class segformer(nn.Module):
    def __init__(self,in_channels, num_classes):
        super().__init__()
        self.backbone = mix_transformer(in_chans=in_channels, embed_dims=(64, 128, 320, 512),
                                    num_heads=(1, 2, 5, 8), depths=(3, 4, 18, 3),
                                    sr_ratios=(8, 4, 2, 1), dropout_p=0.0, drop_path_p=0.1)
        self.decoder_head = segformer_head(in_channels=(64, 128, 320, 512),
                                    num_classes=num_classes, embed_dim=256)

        
    def run(self,image): 
        image_hw = image.shape[2:]
        x = self.backbone(image) #: Call Encoder
        x = self.decoder_head(x) #: Call Decoder
        x = F.interpolate(x, size=image_hw, mode='bilinear', align_corners=False) # Interpolate to output size
        return x
