import torch
import torchaudio
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase

class PitchShift(AugmentationBase):
    def __init__(self, sample_rate, n_steps, **kwargs):
        self._aug = torchaudio.transforms.PitchShift(sample_rate, n_steps, **kwargs)
    
    def __call__(self, data: Tensor) -> Tensor:
        # print('PitchShift')
        x = data.unsqueeze(1)
        x = self._aug(x)
        return x.squeeze(1)
