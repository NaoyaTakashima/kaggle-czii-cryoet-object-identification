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
import cc3d
from transformers import get_cosine_schedule_with_warmup

from src.models.common import get_model
from src.utils.common import reconstruct_array_gpu, dict_to_df
from src.utils.score import score


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
            coordinates = batch['coordinates']
            self.validation_step_outputs.append(
                (
                    y,
                    coordinates,
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
        coordinates = [x[1] for x in self.validation_step_outputs]
        preds = torch.cat([x[2] for x in self.validation_step_outputs])
        losses = np.array([x[3] for x in self.validation_step_outputs])
        loss = losses.mean()

        # if self.current_epoch % 100 != 0:
        # if loss < self.__best_loss:
        #     torch.save(self.model.state_dict(), "best_model.pth")
        #     print(f"Saved best model {self.__best_loss} -> {loss}")
        #     self.__best_loss = loss
        processed_coordinates = []
        for z_tensor, y_tensor, x_tensor in coordinates:
            # 各tensorは同じ長さを想定
            length = z_tensor.size(0)
            for i in range(length):
                # .item()でPythonの数値に変換してintで整数化
                z = int(z_tensor[i].item())
                y = int(y_tensor[i].item())
                x = int(x_tensor[i].item())
                processed_coordinates.append((z, y, x))

        SCALEs = [0.000025] * 5
        CERTAINTY_THRESHOLD = 0.5

        location_df = []
        
        # predsがすでにtorch.Tensorの形でバッチになっていると仮定
        thresh_probs = preds > CERTAINTY_THRESHOLD
        _, max_classes = thresh_probs.max(dim=1)  # バッチ次元も考慮

        original_shape = (184, 630, 630)
        reconstructed_mask = reconstruct_array_gpu(max_classes, processed_coordinates, original_shape)
        
        location = {}

        val_id = self.cfg.split.valid_ids[0].split('-')[0]
        classes = [1, 3, 4, 5, 6]
        id_to_name = {
            1: "apo-ferritin", 
            2: "beta-amylase",
            3: "beta-galactosidase", 
            4: "ribosome", 
            5: "thyroglobulin", 
            6: "virus-like-particle"
        }
        particle_radius = {
            'apo-ferritin': 60,
            'beta-amylase': 65,
            'beta-galactosidase': 90,
            'ribosome': 150,
            'thyroglobulin': 130,
            'virus-like-particle': 135,
        }
        for c in classes:
            cc = cc3d.connected_components((reconstructed_mask == c).cpu().numpy())
            stats = cc3d.statistics(cc)
            zyx=stats['centroids'][1:]*10.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
            zyx_large = zyx[stats['voxel_counts'][1:] > 4 * particle_radius[id_to_name[c]]**3 * np.pi * SCALEs[i] / 3]
            xyz =np.ascontiguousarray(zyx_large[:,::-1])

            location[id_to_name[c]] = xyz

            df = dict_to_df(location, val_id)
            location_df.append(df)
        
        location_df = pd.concat(location_df)
        location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))
        location_df.to_csv("oof.csv", index=False)

        csv_path = f"/kaggle/input/solution/solution_{val_id}.csv"
        solution_df = pd.read_csv(csv_path)
        val_score = score(solution_df, location_df, 'row_id', 0.5, 4)
        self.log(f"val_score_{val_id}", val_score, on_step=False, on_epoch=True, logger=True, prog_bar=True, sync_dist=True)

        if val_score > self.__best_score:
            torch.save(self.model.state_dict(), "best_model.pth")
            print(f"Saved best model {self.__best_score} -> {val_score}")
            self.__best_score = val_score

        self.validation_step_outputs.clear()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]