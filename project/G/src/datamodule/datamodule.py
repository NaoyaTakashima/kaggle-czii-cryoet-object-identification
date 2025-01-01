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
                              RandRotate90d, RandGaussianNoiseD, RandAdjustContrastd,
                              RandAffined, RandZoomd, RandGaussianSmoothd)
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.transforms.functional import resize

from src.utils.common import extract_3d_patches_minimal_overlap


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
                label_key="label", spatial_size=[self.cfg.spatial_size] * 3, 
                num_classes=self.cfg.num_classes, 
                num_samples=self.cfg.num_samples
            ),
            # RandRotate90d(keys=["image", "label"], prob=self.cfg.aug.p_rotate90d, spatial_axes=[0, 1]),
            RandRotate90d(keys=["image", "label"], prob=self.cfg.aug.p_rotate90d, spatial_axes=[0, 2]),
            # RandRotate90d(keys=["image", "label"], prob=self.cfg.aug.p_rotate90d, spatial_axes=[1, 2]),
            RandFlipd(keys=["image", "label"], prob=self.cfg.aug.p_flipd, spatial_axis=0),
            # RandFlipd(keys=["image", "label"], prob=self.cfg.aug.p_flipd, spatial_axis=1),
            # RandFlipd(keys=["image", "label"], prob=self.cfg.aug.p_flipd, spatial_axis=2),
            RandGaussianNoiseD(keys=["image"], prob=self.cfg.aug.p_gaussian_noise, mean=0.0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=self.cfg.aug.p_adjust_contrast, gamma=(0.9, 1.1)),
            # RandAffined(
            #     keys=["image", "label"], prob=self.cfg.aug.p_affine, rotate_range=(0.05, 0.05, 0.05), 
            #     shear_range=(0.02, 0.02, 0.02), translate_range=(2, 2, 2), scale_range=(0.05, 0.05, 0.05), mode=("bilinear", "nearest")
            # ),
            # RandZoomd(keys=["label"], prob=self.cfg.aug.p_zoom, min_zoom=cfg.aug.min_zoom, max_zoom=cfg.aug.max_zoom, mode=("nearest")),
            RandGaussianSmoothd(keys=["image"], prob=self.cfg.aug.p_gaussian_smooth, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5))
        ])

        self.original_ids = ['TS_5_4', 'TS_6_4', 'TS_69_2', 'TS_6_6', 'TS_73_6', 'TS_86_3', 'TS_99_9']
        self.original_ids_dict = {id: 1 for id in self.original_ids}

    def train_dataloader(self):
        train_files = []
        for id in self.cfg.split.train_ids:
            train_id = id.split('-')[0]
            train_modality = id.split('-')[1]
            
            if train_id not in self.original_ids_dict:
                data_dir = '/home/naoya/kaggle/czii/input/extra_data/numpy'
                image_path = os.path.join(os.path.join(data_dir, train_modality), f"train_image_{train_id}.npy")
                label_path = os.path.join(os.path.join(data_dir, train_modality), f"train_label_{train_id}.npy")
            else:
                data_dir = self.cfg.dir.data_dir
                image_path = os.path.join(os.path.join(data_dir, train_modality), f"train_image_{train_id}.npy")
                label_path = os.path.join(os.path.join(data_dir, train_modality), f"train_label_{train_id}.npy")

            image = np.load(image_path)
            label = np.load(label_path)

            train_files.append({"image": image, "label": label})
        
        raw_train_ds = CacheDataset(data=train_files, transform=self.non_random_transforms, cache_rate=0.1)
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
            {
            "image": np.load(os.path.join(os.path.join(self.cfg.dir.data_dir, id.split('-')[1]), f"train_image_{id.split('-')[0]}.npy")),
            "label": np.load(os.path.join(os.path.join(self.cfg.dir.data_dir, id.split('-')[1]), f"train_label_{id.split('-')[0]}.npy"))
            }
            for id in self.cfg.split.valid_ids
        ]
        val_images,val_labels = [dcts['image'] for dcts in valid_files],[dcts['label'] for dcts in valid_files]

        patch_size = self.cfg.patch_size
        val_image_patches, val_image_coords = extract_3d_patches_minimal_overlap(val_images, patch_size)
        val_label_patches, val_label_coords = extract_3d_patches_minimal_overlap(val_labels, patch_size)

        val_patched_data = [{"image": img, "label": lbl, "coordinates": coord} for img, lbl, coord in zip(val_image_patches, val_label_patches, val_image_coords)]

        valid_ds = CacheDataset(data=val_patched_data, transform=self.non_random_transforms, cache_rate=1.0)
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.cfg.batch_size_valid,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
