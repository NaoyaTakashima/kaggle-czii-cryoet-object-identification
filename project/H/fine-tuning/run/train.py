import logging
import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint, RichModelSummary,
                                         RichProgressBar)
from pytorch_lightning.loggers import WandbLogger

from src.datamodule.datamodule import CZIIDataModule
from src.modelmodule.modelmodule import CZIIModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)

directory_path = "output"
os.makedirs(directory_path, exist_ok=True)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    # seed_everything(cfg.seed)

    # set_callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project=cfg.project,
        entity="naoya-takashima",
        save_dir="output/wandb_logs"
    )

    datamodule = CZIIDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = CZIIModel(cfg)
    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,
        max_epochs=cfg.epoch,
        max_steps=cfg.epoch * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, model_summary],
        logger=pl_logger,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
