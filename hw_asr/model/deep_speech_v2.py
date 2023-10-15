from typing import Union
from torch import Tensor, nn
import torch
import math

from hw_asr.base import BaseModel

class DeepSpeechV2(BaseModel):
    def __init__(self, n_feats: int, n_class: int, rnn_layers: int, rnn_dropout: float, **batch):
        super().__init__(n_feats, n_class, **batch)

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41,11), stride=(2,2), padding=(20,5))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21,11), stride=(2,1), padding=(10,5))

        # Batch norms
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        # Activasion
        self.activasion = nn.SiLU()

        # Reccurent block
        # FIXME: apply layernormalization between rnn layers
        self.rnn = nn.GRU(
            input_size=self.n_feats_after_conv(input_feats=n_feats), hidden_size=800, 
            num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=rnn_dropout
        )

        # FC layer
        self.fc = nn.Linear(800 * 2, 1600)
        self.projection = nn.Linear(1600, n_class)

    def _shape_after_conv(self, n_feats: torch.tensor, kernel_size: float, padding: float, stride: float, dilation: float):
        # input tensor shape: (N_batch, N_channels, N_freq, N_timesteps)
        return math.floor(
            (n_feats + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )

    def forward(self, spectrogram, **batch) -> dict:
        """
        :param spectrogram: Mel-log-Spectrogram of size (N_batch, N_frequency, N_timesteps)
        """
        # Transform spectrogram to propper size (N_batch, N_channels, N_frequency, N_timesteps)
        x = spectrogram.unsqueeze(1)

        # input: (N_batch, N_channels, N_freq, N_timesteps)
        x = self.conv1(x)
        x = self.activasion(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.activasion(x)
        x = self.bn2(x)
        # output: (N_batch, N_channels, N_freq, N_timesteps)

        # Transform conv output to recurrent blocks
        n_batch, n_channels, n_freq, n_timesteps = x.size()
        x = x.view(n_batch, n_channels * n_freq, n_timesteps).permute(0, 2, 1)
        # Input: (n_batch, n_time, n_feats)
        x, h_n = self.rnn(x)

        # FC and projection
        x = self.fc(x)
        x = self.activasion(x)
        x = self.projection(x)

        return {"logits": x}
    
    def transform_input_lengths(self, input_lengths):
        # input_lengths = self._shape_after_conv(
        #     n_feats=input_lengths, kernel_size=11.0, padding=5.0, stride=2.0, dilation=1.0
        # )
        # input_lengths = self._shape_after_conv(
        #     n_feats=input_lengths, kernel_size=11.0, padding=5.0, stride=1.0, dilation=1.0
        # )
        return input_lengths // 2
    
    def n_feats_after_conv(self, input_feats):
        input_feats = self._shape_after_conv(
            n_feats=input_feats, kernel_size=41.0, padding=20.0, stride=2.0, dilation=1.0
        )
        input_feats = self._shape_after_conv(
            n_feats=input_feats, kernel_size=21.0, padding=10.0, stride=2.0, dilation=1.0
        )
        return input_feats * 32
