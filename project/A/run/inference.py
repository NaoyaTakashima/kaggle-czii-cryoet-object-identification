import os
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datamodule.datamodule import TestDataset
from src.models.common import get_model
from src.utils.common import trace


def load_model(cfg: DictConfig) -> nn.Module:
    model = get_model(cfg)
    # load weights
    if cfg.weight is not None:
        weight_path = (
            Path(cfg.dir.model_dir)
            / cfg.weight["exp_no"]
            / cfg.weight["exp_name"]
            / "single"
            / "best_model.pth"
        )
        model.load_state_dict(torch.load(weight_path))
        print('load weight from "{}"'.format(weight_path))
    return model


def get_test_dataloader(cfg: DictConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    test_dataset = TestDataset(cfg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    loader: DataLoader, model: nn.Module, device: torch.device, use_amp
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()

    preds = []
    masks = []
    for batch in tqdm(loader, desc="inference"):
        x_ray, mask = batch
        x_ray = x_ray.to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(x_ray, None)["logits"]
            preds.append(pred.detach().cpu().numpy())
            masks.append(mask.numpy())

    preds = np.concatenate(preds)
    masks = np.concatenate(masks)

    return preds, masks
    

@hydra.main(config_path="conf", config_name="inference", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        model = load_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        preds, masks = inference(test_dataloader, model, device, use_amp=cfg.use_amp)

    test_ids = cfg.split.test_patient_ids
    preds_copy = preds.copy()
    for test_id, pred in zip(test_ids, preds_copy):

        # 出力を確率値のまま保存
        save_dir_gray = f"{cfg.dir.inference_dir}/{cfg.exp_no}/{cfg.exp_name}/pred_image_gray"
        if not os.path.exists(save_dir_gray):
            os.makedirs(save_dir_gray)
        np.save(f"{save_dir_gray}/{test_id}.npy", pred[0])
        cv2.imwrite(f"{save_dir_gray}/{test_id}.png", pred[0])
        # predを適切な範囲に変換 (例: 0-255)
        pred_image = (pred > 0.5).astype(np.uint8)
        pred_image *= 255

        save_dir = f"{cfg.dir.inference_dir}/{cfg.exp_no}/{cfg.exp_name}/pred_image"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(f"{save_dir}/{test_id}.png", pred_image[0])
    
    masks_tensor = torch.from_numpy(masks).type(torch.int)
    preds_tensor = torch.from_numpy(preds)

    tp, fp, fn, tn = smp.metrics.get_stats(preds_tensor, masks_tensor, mode='binary', threshold=0.5)
    all_tp = torch.sum(tp)
    all_fp = torch.sum(fp)
    all_fn = torch.sum(fn)
    ALL_DSC = 2*all_tp / (2*all_tp + all_fp + all_fn)
    scores = smp.metrics.f1_score(tp, fp, fn, tn)
    
    tp_1d = tp.flatten()
    fp_1d = fp.flatten()
    fn_1d = fn.flatten()
    tn_1d = tn.flatten()
    scores_1d = scores.flatten()
    
    df = pd.DataFrame({
        'id': test_ids,
        'tp': tp_1d,
        'fp': fp_1d,
        'fn': fn_1d,
        'tn': tn_1d,
        'DICE': scores_1d,
    })
    df.to_csv(f"{cfg.dir.inference_dir}/{cfg.exp_no}/{cfg.exp_name}/summary.csv", index=False)
    
    print(f"ALL_DSC : {ALL_DSC}")
    print(f"MEAN_DSC : {scores.mean()}")


if __name__ == "__main__":
    main()
