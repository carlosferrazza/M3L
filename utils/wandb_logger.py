from typing import  List, Optional

import torch
import wandb
import numpy as np
from stable_baselines3.common.logger import Logger, KVWriter
from stable_baselines3.common.logger import Image, Video


class WandbLogger(Logger):
    def __init__(self, folder: Optional[str], output_formats: List[KVWriter], log_interval: int = 1000):
        super().__init__(folder, output_formats)
        self.log_interval = log_interval
        self.i_log = 0

    def dump(self, step: int = 0) -> None:
        if step - self.i_log >= self.log_interval:
            convert_name = {k: k for k in self.name_to_value.keys()}
            convert_name.update({  # SB3 <-> wandb
                'rollout/ep_len_mean': 'rollout/ep_len_mean',
                'rollout/ep_rew_mean': 'rollout/ep_rew_mean',
                'time/fps': 'time/fps',
                'train/approx_kl': 'train/approx_kl',
                'train/clip_fraction': 'train/clip_fraction',
                'train/clip_range': 'train/clip_range',
                'train/entropy_loss': 'train/entropy_loss',
                'train/explained_variance': 'train/explained_variance',
                'train/learning_rate': 'train/learning_rate',
                'train/loss': 'train/loss',
                'train/policy_gradient_loss': 'train/policy_gradient_loss',
                'train/std': 'train/std',
                'train/value_loss': 'train/value_loss'
            })

            log_result = {convert_name[k]: v for k, v in self.name_to_value.items()}

            for k, v in self.name_to_value.items():
                if isinstance(v, Video):
                    if isinstance(v.frames, torch.Tensor):
                        v.frames = v.frames.numpy()
                    if isinstance(v.frames, np.ndarray) and v.frames.dtype != np.uint8:
                        v.frames = (255 * np.clip(v.frames, 0, 1)).astype(np.uint8)
                    log_result[k] = wandb.Video(v.frames, fps=v.fps, format="gif")

            wandb.log(log_result, step)
            self.i_log = step

            super().dump(step)
