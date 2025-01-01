from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from src.models.common import get_model
from src.utils.common import (PARTICLE, dotdict, location_to_df,
                              probability_to_location, read_one_truth)
from src.utils.score import score
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
        prob = output["particle"]
        # print(prob.shape)

        if mode == 'val':
            self.validation_step_outputs.append(
                (
                    y,
                    prob.detach(),
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
        prob = torch.cat([x[1] for x in self.validation_step_outputs])
        preds = torch.cat([x[2] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        # print(prob.shape)
        # print(y.shape, preds.shape)
        cfg = dotdict(
            threshold={ 
                'apo-ferritin': 0.05,
                'beta-amylase': 0.05,
                'beta-galactosidase': 0.05,
                'ribosome': 0.05,
                'thyroglobulin': 0.05,
                'virus-like-particle': 0.05,
            },
        )


        if loss < self.__best_loss:
            torch.save(self.model.state_dict(), "best_model.pth")
            print(f"Saved best model {self.__best_loss} -> {loss}")
            self.__best_loss = loss

        D, H, W = 184, 630, 630
        num_slice = self.cfg.num_slice
        submit_df = []
        for i, id in enumerate(self.cfg.split.valid_ids):
            probability = np.zeros((7, D, H, W), dtype=np.float32)
            count = np.zeros((7, D, H, W), dtype=np.float32)

            zz = list(range(0, D - num_slice, num_slice//2)) + [D - num_slice]
            for i, z in enumerate(zz):
                prob_ = prob[i].cpu().numpy()
                # print(prob_.shape)
                probability[:, z:z + num_slice, ...] += prob_[:, :, :H, :W]
                count[:, z:z + num_slice, ...] += 1
            
            probability = probability / (count + 0.0001)
            location = probability_to_location(probability, cfg)
            df = location_to_df(location)
            df.insert(loc=0, column='experiment', value=id)
            submit_df.append(df)

            solution_df = pd.read_csv(f"/home/naoya/kaggle/czii/input/solution/solution_{id}.csv")
        
            submit_df = pd.concat(submit_df)
            submit_df.insert(loc=0, column='id', value=np.arange(len(submit_df)))
            submit_df.to_csv("oof.csv", index=False)

            val_score = score(solution_df, submit_df, 'row_id', 0.5, 4)
            self.log(f"val_score_{id}", val_score, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]