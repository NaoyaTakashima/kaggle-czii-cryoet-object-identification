from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceFocalLoss, DiceLoss, TverskyLoss
from monai.networks.nets import UNet


class Unet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Union[Tuple[int, ...], List[int]],
        strides: Union[Tuple[int, ...], List[int]],
        num_res_units: int,
    ):
        super().__init__()
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )
        
        self.class_weights = torch.tensor([1, 1, 0, 2, 1, 2, 1])
        self.loss_fn = TverskyLoss(include_background=True, to_onehot_y=True, softmax=True, class_weight=self.class_weights)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        logits = self.model(x)

        output = {"logits": logits}
        if y is not None:
            loss = self.loss_fn(logits, y)
            output["loss"] = loss

        return output