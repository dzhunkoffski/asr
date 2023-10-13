import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase

class FrequencyMasking(AugmentationBase):
    def __init__(self, freq_mask_param: int, **kwargs):
        self._aug = torchaudio.transforms.FrequencyMasking(freq_mask_param)
    
    def __call__(self, data: Tensor):
        x = self._aug(data)
        return x