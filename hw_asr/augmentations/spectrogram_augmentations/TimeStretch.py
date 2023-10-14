import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase
import random

class TimeStretch(AugmentationBase):
    def __init__(self, p: float, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)
        self.p = p
    
    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = data
            x = self._aug(x)
            return x
        return data
