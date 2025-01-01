import os
from glob import glob

import numpy as np
import torch
# import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.transforms import (AsDiscrete, Compose, EnsureChannelFirstd,
                              NormalizeIntensityd, Orientationd,
                              RandCropByLabelClassesd, RandFlipd,
                              RandRotate90d)
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from src.utils.common import extract_3d_patches_minimal_overlap
from torchvision import transforms
from torchvision.transforms.functional import resize


class CZIIDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.non_random_transforms = Compose([
            EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
            NormalizeIntensityd(keys="image"),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ])
        self.random_transforms = Compose([
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[self.cfg.spatial_size, self.cfg.spatial_size, self.cfg.spatial_size],
                num_classes=self.cfg.num_classes,
                num_samples=self.cfg.num_samples
            ),
            RandRotate90d(keys=["image", "label"], prob=self.cfg.aug.p_rotate90d, spatial_axes=[0, 2]),
            RandFlipd(keys=["image", "label"], prob=self.cfg.aug.p_flipd, spatial_axis=0),
        ])

    def train_dataloader(self):
        train_files = [
            {"image": np.load(os.path.join(self.cfg.dir.data_dir, f"train_image_{id}.npy")),
             "label": np.load(os.path.join(self.cfg.dir.data_dir, f"train_label_{id}.npy"))}
            for id in self.cfg.split.train_ids
        ]
        raw_train_ds = CacheDataset(data=train_files, transform=self.non_random_transforms, cache_rate=1.0)
        train_ds = Dataset(data=raw_train_ds, transform=self.random_transforms)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size_train,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_files = [
            {"image": np.load(os.path.join(self.cfg.dir.data_dir, f"train_image_{id}.npy")),
             "label": np.load(os.path.join(self.cfg.dir.data_dir, f"train_label_{id}.npy"))}
            for id in self.cfg.split.valid_ids
        ]
        val_images,val_labels = [dcts['image'] for dcts in valid_files],[dcts['label'] for dcts in valid_files]

        val_image_patches, _ = extract_3d_patches_minimal_overlap(val_images, self.cfg.patch_size)
        val_label_patches, _ = extract_3d_patches_minimal_overlap(val_labels, self.cfg.patch_size)

        val_patched_data = [{"image": img, "label": lbl} for img, lbl in zip(val_image_patches, val_label_patches)]

        valid_ds = CacheDataset(data=val_patched_data, transform=self.non_random_transforms, cache_rate=1.0)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size_valid,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
