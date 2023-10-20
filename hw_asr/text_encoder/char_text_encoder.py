import json
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

import numpy as np
from torch import Tensor

from hw_asr.base.base_text_encoder import BaseTextEncoder

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
import logging
import glob

TOKENIZER_PATH = 'hw_asr/text_encoder/tokenizers'

logger = logging.getLogger(__name__)


class CharTextEncoder(BaseTextEncoder):

    def __init__(self, tokenizer: str = 'char', alphabet: List[str] = None, tokenizer_config: dict = None):
        assert tokenizer in ['char', 'bpe'], "Unkown tokenizer"
        self.tokenizer_policy = tokenizer
        self.tokenizer = None
        if alphabet is None and tokenizer == 'char':
            alphabet = self._load_char_vocab()
        elif alphabet is None and tokenizer == 'bpe':
            self.tokeinzer, alphabet = self._load_bpe_vocab(tokenizer_config)
        self.alphabet = alphabet
        self.ind2char = {k: v for k, v in enumerate(sorted(alphabet))}
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def _load_char_vocab(self):
        return list(ascii_lowercase + ' ')
    
    def _load_train_text_paths(self, config):
        corpus_paths = []
        logger.info(f'Tokenizer trains on {config["train_texts"]} datasets')
        for train_corpus in config['train_texts']:
            # train_corpus looks smth like `librispeech/train-clean-100`
            corpus_paths += glob.glob(f'data/datasets/{train_corpus}/**.txt')
        logger.info(f'Tokenizer trains on {len(corpus_paths)} texts')
        return corpus_paths

    def _load_bpe_vocab(self, config):
        tok_path = f'{TOKENIZER_PATH}/bpe__vocab_{config["vocab_size"]}__minfreq_{config["min_frequency"]}__maxlen_{config["max_token_length"]}.json'
        if os.path.exists(tok_path):
            logger.info('BPE tokenizer found, loading pretrain')
            tokenizer = Tokenizer.from_file(tok_path)
        else:
            logger.info('Train new BPR tokenizer')
            tokenizer = Tokenizer(BPE())
            trainer = BpeTrainer(
                vocab_size=config['vocab_size'],
                min_frequency=config['min_frequency'],
                show_progress =config['show_progress'],
                max_token_length=config['max_token_length']
            )
            tokenizer.pre_tokenizer = Whitespace()
            tokenizer.train(self._load_train_text_paths(config), trainer)
            tokenizer.save(tok_path)    
        vocab = list(tokenizer.get_vocab().keys())
        vocab = [self.normalize_text(token) for token in vocab]
        return tokenizer, vocab + [' ']

    def __len__(self):
        return len(self.ind2char)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> Tensor:
        text = self.normalize_text(text)
        if self.tokenizer_policy == 'char':
            return self._encode_char(text)
        if self.tokenizer_policy == 'bpe':
            return self._encoder_bpe(text)

    def _encode_char(self, text):
        try:
            return Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError as e:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'")
    
    def _encoder_bpe(self, text):
        enc = self.tokeinzer.encode(text)
        try:
            return Tensor([self.char2ind[token] for token in enc.tokens]).unsqueeze(0)
        except KeyError as e:
            unknown_toks = set([tok for tok in enc.tokens if tok not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown tokens: '{' '.join(unknown_toks)}'")
            

    def decode(self, vector: Union[Tensor, np.ndarray, List[int]]):
        return ''.join([self.ind2char[int(ind)] for ind in vector]).strip()

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
