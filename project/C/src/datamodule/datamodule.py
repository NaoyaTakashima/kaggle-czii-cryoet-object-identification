import os
from glob import glob

import numpy as np
import torch
# import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.data import (CacheDataset, DataLoader, Dataset, decollate_batch,
                        pad_list_data_collate)
from monai.transforms import (AsDiscrete, Compose, EnsureChannelFirstd,
                              NormalizeIntensityd, Orientationd,
                              RandCropByLabelClassesd, RandFlipd,
                              RandRotate90d, Resized)
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from src.utils.common import pad_hw, preprocess_volume_with_pad
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
            # Resized(keys=["image", "label"], spatial_size=(184, self.cfg.img_size, self.cfg.img_size), mode="nearest") # 計算負荷が非常に大きい
        ])
        self.random_transforms = Compose([
            RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[self.cfg.num_slice, self.cfg.img_size, self.cfg.img_size],
                num_classes=self.cfg.num_classes,
                num_samples=self.cfg.num_samples
            ),
            RandRotate90d(keys=["image", "label"], prob=self.cfg.aug.p_rotate90d, spatial_axes=[1, 2]),
            RandFlipd(keys=["image", "label"], prob=self.cfg.aug.p_flipd, spatial_axis=0),
        ])

    def train_dataloader(self):
        train_files = []
        for id in self.cfg.split.train_ids:
            # ImageとLabelの読み込み
            image_volume = np.load(os.path.join(self.cfg.dir.data_dir, f"train_image_{id}.npy"))
            label_volume = np.load(os.path.join(self.cfg.dir.data_dir, f"train_label_{id}.npy"))

            # パディングとスライスの適用
            image_volume = pad_hw(image_volume, self.cfg.img_size)
            label_volume = pad_hw(label_volume, self.cfg.img_size)

            train_files.append({"image": image_volume, "label": label_volume})
        
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
        valid_files = []
        for id in self.cfg.split.valid_ids:
            # ImageとLabelの読み込み
            image_volume = np.load(os.path.join(self.cfg.dir.data_dir, f"train_image_{id}.npy"))
            label_volume = np.load(os.path.join(self.cfg.dir.data_dir, f"train_label_{id}.npy"))

            # パディングとスライスの適用
            image_slices = preprocess_volume_with_pad(image_volume, self.cfg.num_slice, self.cfg.img_size)
            label_slices = preprocess_volume_with_pad(label_volume, self.cfg.num_slice, self.cfg.img_size)

            # スライス単位でデータを構築
            for img_slice, lbl_slice in zip(image_slices, label_slices):
                valid_files.append({"image": img_slice, "label": lbl_slice})

        valid_ds = CacheDataset(data=valid_files, transform=self.non_random_transforms, cache_rate=1.0)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size_valid,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
