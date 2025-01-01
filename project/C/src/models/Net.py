from typing import List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, TverskyLoss
from src.models.decoder import *

#from timm.models.convnext import *


#------------------------------------------------
# processing

def encode_for_resnet(e, x, B, depth_scaling=[2,2,2,2,1]):

    def pool_in_depth(x, depth_scaling):
        bd, c, h, w = x.shape
        x1 = x.reshape(B, -1, c, h, w).permute(0, 2, 1, 3, 4)
        x1 = F.avg_pool3d(x1, kernel_size=(depth_scaling, 1, 1), stride=(depth_scaling, 1, 1), padding=0)
        x = x1.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        return x, x1

    encode=[]
    x = e.conv1(x)
    x = e.bn1(x)
    x = e.act1(x)
    x, x1 = pool_in_depth(x, depth_scaling[0])
    encode.append(x1)
    #print(x.shape)

    #x = e.maxpool(x)
    x = F.avg_pool2d(x,kernel_size=2,stride=2)

    x = e.layer1(x)
    x, x1 = pool_in_depth(x, depth_scaling[1])
    encode.append(x1)
    #print(x.shape)

    x = e.layer2(x)
    x, x1 = pool_in_depth(x, depth_scaling[2])
    encode.append(x1)
    #print(x.shape)

    x = e.layer3(x)
    x, x1 = pool_in_depth(x, depth_scaling[3])
    encode.append(x1)
    #print(x.shape)

    x = e.layer4(x)
    x, x1 = pool_in_depth(x, depth_scaling[4])
    encode.append(x1)
    #print(x.shape)

    return encode

class Net(nn.Module):
    def __init__(
            self, 
            pretrained: bool,
            num_classes: int,
            model_name: str,
        ):

        super().__init__()

        encoder_dim = {
            'resnet18': [64, 64, 128, 256, 512, ],
            'resnet18d': [64, 64, 128, 256, 512, ],
            'resnet34d': [64, 64, 128, 256, 512, ],
            'resnet50d': [64, 256, 512, 1024, 2048, ],
            'seresnext26d_32x4d': [64, 256, 512, 1024, 2048, ],
            'convnext_small.fb_in22k': [96, 192, 384, 768],
            'convnext_tiny.fb_in22k': [96, 192, 384, 768],
            'convnext_base.fb_in22k': [128, 256, 512, 1024],
            'tf_efficientnet_b4.ns_jft_in1k':[32, 56, 160, 448],
            'tf_efficientnet_b5.ns_jft_in1k':[40, 64, 176, 512],
            'tf_efficientnet_b6.ns_jft_in1k':[40, 72, 200, 576],
            'tf_efficientnet_b7.ns_jft_in1k':[48, 80, 224, 640],
            'pvt_v2_b1': [64, 128, 320, 512],
            'pvt_v2_b2': [64, 128, 320, 512],
            'pvt_v2_b4': [64, 128, 320, 512],
        }.get(model_name, [768])
        decoder_dim = \
            [256, 128, 64, 32, 16] #   [384, 192, 96, 32, 16]

        self.encoder = timm.create_model(
            model_name=model_name, pretrained=pretrained, in_chans=3, num_classes=0, global_pool='', features_only=True,
        )
        self.decoder = MyUnetDecoder3d(
            in_channel=encoder_dim[-1],
            skip_channel=encoder_dim[:-1][::-1]+[0],
            out_channel=decoder_dim,
        )
        self.mask = nn.Conv3d(decoder_dim[-1], num_classes, kernel_size=1)

        self.loss_fn_tv = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True)
        


    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        B, C, D, H, W = x.shape

        x = x.reshape(B*D, C, H, W)

        x = (x.float() - 0.5) / 0.5
        x = x.expand(-1, 3, -1, -1)

        encode = self.encoder(x)[-5:]
        encode = encode_for_resnet(self.encoder, x, B, depth_scaling=[2,2,2,2,1])
        # [print(f'encode_{i}', e.shape) for i,e in enumerate(encode)]

        # [print(f'encode_{i}', e.shape) for i, e in enumerate(encode)]
        last, decode = self.decoder(
            feature=encode[-1], skip=encode[:-1][::-1]+[None], depth_scaling=[1,2,2,2,2]
        )
        # print(f'last', last.shape)

        logits = self.mask(last)
        # print('logit', logits.shape)

        # print(logits.shape, y.shape)
        # print(y.shape)

        output = {"logits": logits}
        output['particle'] = F.softmax(logits, 1)
        if y is not None:
            # loss = self.loss_fn_tv(logits, y)
            loss = F.cross_entropy(logits, y.squeeze(1).long())
            output["loss"] = loss

        return output
