from typing import List

import torch
from torch import Tensor

from tqdm.autonotebook import tqdm

from hw_asr.base.base_metric import BaseMetric
from hw_asr.base.base_text_encoder import BaseTextEncoder
from hw_asr.metric.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = BaseTextEncoder.normalize_text(target_text)
            if hasattr(self.text_encoder, "ctc_decode"):
                pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            else:
                pred_text = self.text_encoder.decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

class LMBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, epoch_freq: int, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.last_value = 1.0
        self.epoch_freq = epoch_freq
        self.beam_size = beam_size
    
    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], epoch: int, **kwargs):
        wers = []
        if epoch % self.epoch_freq == 0:
            for log_prob_vec, length, target_text in zip(log_probs, log_probs_length, text):
                target_text = BaseTextEncoder.normalize_text(target_text)
                pred_text = self.text_encoder.ctc_beam_search_with_lm(log_prob_vec, length, beam_size=self.beam_size)[0].text
                wers.append(calc_wer(target_text, pred_text))
            self.last_value = sum(wers) / len(wers)
        return self.last_value

class NoLMBeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder: BaseTextEncoder, epoch_freq: int, beam_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.last_value = 1.0
        self.epoch_freq = epoch_freq
        self.beam_size = beam_size
    
    def __call__(self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], epoch: int, **kwargs):
        wers = []
        if epoch % self.epoch_freq == 0:
            for log_prob_vec, length, target_text in zip(log_probs, log_probs_length, text):
                target_text = BaseTextEncoder.normalize_text(target_text)
                pred_text = self.text_encoder.ctc_beam_search_without_lm(log_prob_vec, length, beam_size=self.beam_size)[0].text
                wers.append(calc_wer(target_text, pred_text))
            self.last_value = sum(wers) / len(wers)
        return self.last_value
