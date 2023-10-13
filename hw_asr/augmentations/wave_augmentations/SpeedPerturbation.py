from torch import Tensor
import torchaudio

from hw_asr.augmentations.base import AugmentationBase

class SpeedPerturbation(AugmentationBase):
    def __init__(self, *args, **kwargs):
        self._aug = torchaudio.transforms.SpeedPerturbation(*args, **kwargs)
    
    def __call__(self, data: Tensor):
        # print('SpeedPerturbation')
        x = data.unsqueeze(1)
        x, _ = self._aug(x)
        return x.squeeze(1)
