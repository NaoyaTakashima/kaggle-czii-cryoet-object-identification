from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch_3d as smp
from segmentation_models_pytorch_3d.losses import DiceLoss, TverskyLoss

class Unet_smp(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        classes: int,
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )
        
        self.class_weights = torch.tensor([1, 1, 0, 2, 1, 2, 1])

        self.loss_fn_tv = TverskyLoss(mode="multiclass", from_logits=True)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        logits = self.model(x)
        # print(logits.shape)

        output = {"logits": logits}
        if y is not None:
            loss_tv = self.loss_fn_tv(logits, y.long())

            loss = loss_tv
            output["loss"] = loss

        return output