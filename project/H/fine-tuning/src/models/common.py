from typing import Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.Unet import Unet

MODELS = Union[Unet]


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
    else:
        raise NotImplementedError
    
    if cfg.pre_training == True:
        model.load_state_dict(torch.load(cfg.weight_path))
        print('load weight from "{}"'.format(cfg.weight_path))

    return model