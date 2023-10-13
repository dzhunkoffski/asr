import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase

class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.TimeStretch(*args, **kwargs)
    
    def __call__(self, data: Tensor):
        x = data
        x = self._aug(x)
        return x
