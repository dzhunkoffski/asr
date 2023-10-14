import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase
import random

class TimeMasking(AugmentationBase):
    def __init__(self, p: float, time_mask_param: int, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.p = p
    
    def __call__(self, data: Tensor):
        if random.random() < self.p:
            x = self._aug(data)
            return x
        return data