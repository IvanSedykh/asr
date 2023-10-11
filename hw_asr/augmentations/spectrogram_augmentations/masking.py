import torch
from torch import Tensor
from torchaudio import transforms as T

from hw_asr.augmentations.base import AugmentationBase


class TimeMasking(AugmentationBase):
    def __init__(self, n_masks=1, *args, **kwargs):
        self.n_masks = n_masks
        self._aug = T.TimeMasking(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        for _ in range(self.n_masks):
            x = self._aug(x)
        return x.squeeze(1)


class FrequencyMasking(AugmentationBase):
    def __init__(self, n_masks=1, *args, **kwargs):
        self.n_masks = n_masks
        self._aug = T.FrequencyMasking(*args, **kwargs)

    @torch.no_grad()
    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        for _ in range(self.n_masks):
            x = self._aug(x)
        return x.squeeze(1)
