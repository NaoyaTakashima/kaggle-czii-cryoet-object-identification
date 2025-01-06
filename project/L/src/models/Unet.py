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
        alpha: float,
        beta: float,
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
        
        self.class_weights = torch.tensor([1/113, 16/113, 0, 32/113, 16/113, 32/113, 16/113])

        # self.loss_fn = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True)
        self.loss_fn_tv = TverskyLoss(include_background=False, to_onehot_y=False, sigmoid=True, alpha=alpha, beta=beta)
        self.loss_fn_dice = DiceLoss(include_background=True, to_onehot_y=False, softmax=True)
        self.loss_fn_bce = nn.BCEWithLogitsLoss()
        # self.loss_fn = DiceFocalLoss(include_background=True, to_onehot_y=True, softmax=True, gamma=2.0, weight=class_weights)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        logits = self.model(x)
        # print(logits.shape)

        output = {"logits": logits}
        if y is not None:
            # print(logits.shape, y.shape)
            # print(y.shape)
            # loss_bce = self.loss_fn_bce(logits, y)
            # loss_dice = self.loss_fn_dice(logits, y)
            loss_tv = self.loss_fn_tv(logits, y)
            # クラスごとの損失計算
            

            # loss = 0.3*loss_ce + 0.3*loss_dice + 0.4*loss_tv
            loss = loss_tv
            # loss = loss_dice
            # loss = loss_bce
            output["loss"] = loss

        return output