from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from src.models.common import get_model
from transformers import get_cosine_schedule_with_warmup


class CZIIModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(cfg)
        self.validation_step_outputs: list = []
        self.metric_fn = DiceMetric(include_background=False, reduction="mean", ignore_empty=True)

        self.__best_loss = np.inf
        self.__best_score = -1.0

    def forward(
            self, x: torch.Tensor, y: Optional[torch.Tensor]
        ) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, 'val')

    def __share_step(self, batch, mode: str) -> torch.Tensor:
        x, y = batch['image'], batch['label']
        output = self.model(x, y)
        loss: torch.Tensor = output["loss"]
        logits = output["logits"]

        if mode == 'val':
            self.validation_step_outputs.append(
                (
                    y,
                    logits.detach(),
                    loss.detach().item(),
                )
            )

        self.log(
            f'{mode}_loss',
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_end(self):
        y = torch.cat([x[0] for x in self.validation_step_outputs])
        preds = torch.cat([x[1] for x in self.validation_step_outputs])
        losses = np.array([x[2] for x in self.validation_step_outputs])
        loss = losses.mean()

        # print(y.shape)
        # print(preds.shape)

        metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.cfg.model.out_channels)(i) for i in decollate_batch(preds)]
        metric_val_labels = [AsDiscrete(to_onehot=self.cfg.model.out_channels)(i) for i in decollate_batch(y)]
        self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
        metrics = self.metric_fn.aggregate(reduction="mean_batch")
        val_metric = torch.mean(metrics)
        self.log("val_metric", val_metric, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        if loss < self.__best_loss:
            torch.save(self.model.state_dict(), "best_model.pth")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_loss = loss
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]