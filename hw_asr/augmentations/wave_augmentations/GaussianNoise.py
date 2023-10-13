import torch
from torch import Tensor
from hw_asr.augmentations.base import AugmentationBase

class GaussianNoise(AugmentationBase):
    def __init__(self, mean, std, **kwargs):
        self.noiser = torch.distributions.normal.Normal(mean, std)
    
    def __call__(self, data: Tensor) -> Tensor:
        # print('GaussianNoise')
        x = data.unsqueeze(1)
        x = x + self.noiser.sample(x.shape)
        x = x.squeeze(1)
        return x
