import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase

class TimeMasking(AugmentationBase):
    def __init__(self, time_mask_param: int, **kwargs):
        self._aug = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
    
    def __call__(self, data: Tensor):
        x = self._aug(data)
        return x