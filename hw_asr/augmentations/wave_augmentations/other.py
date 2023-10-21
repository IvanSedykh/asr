import torch
from torch import Tensor
from torchaudio import transforms as T

from hw_asr.augmentations.base import AugmentationBase


class AddNoise(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = T.AddNoise(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        x = self._aug(x)
        return x.squeeze(1)


class SpeedPerturbation(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = T.SpeedPerturbation(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        x = self._aug(x)
        return x.squeeze(1)
