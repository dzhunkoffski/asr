import logging
from typing import List
import torch

logger = logging.getLogger(__name__)
PADDING_VALUE = -1

def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    audio_batch = []
    spectrogram_batch = []
    duration_batch = []
    text_batch = []
    text_encoded_batch = []
    text_encoded_length_batch = []
    spectrogram_length_batch = []
    max_spectrogram_len = 0
    for item in dataset_items:
        audio_batch.append(item['audio'])
        spectrogram_length_batch.append(item['spectrogram'].size()[-1])
        max_spectrogram_len = max(item['spectrogram'].size()[-1], max_spectrogram_len)
        duration_batch.append(item['duration'])
        text_batch.append(item['text'])
        text_encoded_batch.append(item['text_encoded'].squeeze(0))
        text_encoded_length_batch.append(item['text_encoded'].size()[1])
    for item in dataset_items:
        spectrogram_batch.append(
            torch.nn.functional.pad(input=item['spectrogram'], pad=(0, max_spectrogram_len - item['spectrogram'].size()[-1]), mode='constant', value=0).squeeze(0)
        )
    
    # Padding and converting to tensor
    spectrogram_batch = torch.stack(spectrogram_batch)
    text_encoded_batch = torch.nn.utils.rnn.pad_sequence(text_encoded_batch, batch_first=True, padding_value=PADDING_VALUE).int()
    text_encoded_length_batch = torch.tensor(text_encoded_length_batch).int()
    spectrogram_length_batch = torch.tensor(spectrogram_length_batch).int()
    return {
        'audio': audio_batch,
        'spectrogram': spectrogram_batch,
        'duration': duration_batch,
        'text': text_batch,
        'text_encoded': text_encoded_batch,
        'text_encoded_length': text_encoded_length_batch,
        'spectrogram_length': spectrogram_length_batch
    }