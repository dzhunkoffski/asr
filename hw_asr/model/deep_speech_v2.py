from typing import Union
from torch import Tensor, nn
import torch
import math

from hw_asr.base import BaseModel

class RNN_SeqNorm_Block(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, batch_first: bool, bidirectional: bool):
        super().__init__()
        self.num_layers = num_layers
        self.rnn = nn.ModuleList(
            [nn.GRU(
                input_size=input_size, hidden_size=hidden_size, 
                num_layers=1, dropout=dropout, batch_first=batch_first, bidirectional=bidirectional
            )] + [nn.GRU(
                input_size=2 * hidden_size, hidden_size=hidden_size, 
                num_layers=1, dropout=dropout, batch_first=batch_first, bidirectional=bidirectional
                ) for _ in range(num_layers - 1)]
        )
        self.ln = nn.ModuleList(
            [nn.LayerNorm(2 * hidden_size) for _ in range(num_layers)]
        )
    
    def forward(self, x):
        x, h = self.rnn[0](x)
        x = self.ln[0](x)
        for layer_ix in range(1, self.num_layers):
            x, h = self.rnn[layer_ix](x, h)
            x = self.ln[layer_ix](x)
        return x, h

class DeepSpeechV2(BaseModel):
    def __init__(self, n_feats: int, n_class: int, rnn_layers: int, rnn_dropout: float, conv_dropout: float, rnn_normalization: bool, **batch):
        super().__init__(n_feats, n_class, **batch)

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41,11), stride=(2,2), padding=(20,5))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21,11), stride=(2,1), padding=(10,5))

        # Batch norms
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        # Dropout
        self.dropout = nn.Dropout2d(p=conv_dropout)

        # Activasion
        self.activasion = nn.Hardtanh(min_val=0, max_val=20)

        # Reccurent block
        # FIXME: apply layernormalization between rnn layers
        if not rnn_normalization:
            self.rnn = nn.GRU(
                input_size=self.n_feats_after_conv(input_feats=n_feats), hidden_size=800, 
                num_layers=rnn_layers, batch_first=True, bidirectional=True, dropout=rnn_dropout
            )
        else:
            self.rnn = RNN_SeqNorm_Block(
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
        x = self.dropout(x)
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
