from typing import Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.Unet import Unet
from src.models.Unet_smp import Unet_smp

MODELS = Union[Unet, Unet_smp]


def get_model(cfg: DictConfig) -> MODELS:
    model: MODELS

    if cfg.model.name == "Unet":
        model = Unet(
            spatial_dims=cfg.model.spatial_dims,
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            channels=cfg.model.channels,
            strides=cfg.model.strides,
            num_res_units=cfg.model.num_res_units,
            alpha=cfg.model.alpha,
            beta=cfg.model.beta,
        )
    elif cfg.model.name == "Unet_smp":
        model = Unet_smp(
            encoder_name=cfg.model.encoder_name,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.classes,
        )
    else:
        raise NotImplementedError

    return model