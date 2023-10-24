import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor

from hw_asr.base.base_text_encoder import BaseTextEncoder
import sentencepiece as spm

import os
import logging
import glob

TOKENIZER_PATH = 'hw_asr/text_encoder/tokenizers'

logger = logging.getLogger(__name__)


class CharTextEncoder(BaseTextEncoder):

    def __init__(self, alphabet: List[str] = None, tokenizer_name: str = 'char', tokenizer_config: dict = None):
        # if alphabet is None:
        #     alphabet = list(ascii_lowercase + ' ')
        assert(tokenizer_name in ['char', 'bpe'])
        self.tokenizer = None
        self.tokenizer_name = tokenizer_name
        alphabet = self._load_alphabet(tokenizer_name, tokenizer_config)
        self.alphabet = alphabet
        self.ind2char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def _prepare_training_texts(self, train_datasets: List[str]):
        train_texts = []
        train_texts_paths = []
        for train_dataset in train_datasets:
            train_texts_paths += glob.glob(f'{train_dataset}/**/*.txt', recursive=True)
        
        train_corpus = []
        for train_text_path in train_texts_paths:
            with open(train_text_path, 'r') as fd:
                for text in fd.readlines():
                    train_corpus.append(self.normalize_text(text))
        
        with open('train.txt', 'w') as fd:
            for line in train_corpus:
                fd.write(f"{line[1:]}\n")

    
    def _load_alphabet(self, tokenizer_name: str, tokenizer_config: dict = None):
        if tokenizer_name == 'char':
            return list(ascii_lowercase + ' ')
        
        # else bpe
        if tokenizer_config['model_path'] is not None and os.path.exists(tokenizer_config['model_path'] + '.model'):
            self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_config['model_path'] + '.model')
            return  [self.tokenizer.id_to_piece(i) for i in range(self.tokenizer.get_piece_size())]
        else:
            if 'train_text_path' not in tokenizer_config:
                self._prepare_training_texts(tokenizer_config['train_datasets'])
            spm.SentencePieceTrainer.train(f'--input={tokenizer_config["train_text_path"]} --model_prefix={tokenizer_config["model_path"]} --vocab_size={tokenizer_config["vocab_size"]} --model_type=bpe --unk_id=0 --bos_id=-1 --eos_id=-1')
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(tokenizer_config['model_path'] + '.model')
            return  [self.tokenizer.id_to_piece(i) for i in range(self.tokenizer.get_piece_size())]
        


    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        if self.tokenizer_name == 'char':
            return self._encode_char(text)
        return self._encode_bpe(text)

    def _encode_char(self, text) -> Tensor:
        try:
            return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")
        
    def _encode_bpe(self, text) -> Tensor:
        return Tensor(self.tokenizer.encode_as_ids(text)).unsqueeze(0) + 1

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)].replace('▁', '') for ind in vector]).strip()

    def dump(self, file):
        with Path(file).open('w') as f:
            json.dump(self.ind2char, f)

    @classmethod
    def from_file(cls, file):
        with Path(file).open() as f:
            ind2char = json.load(f)
        a = cls([])
        a.ind2char = ind2char
        a.char2ind = {v: k for k, v in ind2char}
        return a
