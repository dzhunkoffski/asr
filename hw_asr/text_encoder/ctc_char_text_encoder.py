from typing import List, NamedTuple
from collections import defaultdict

from pyctcdecode import build_ctcdecoder

import torch

from .char_text_encoder import CharTextEncoder

class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None, lm_n_gram: str = None, lm_prun_thr: str = None):
        # FIXME: load lm via script
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}
        if lm_n_gram is not None and lm_prun_thr is not None:
            print('with lm')
            self.decoder = build_ctcdecoder(
                labels=[''] + list(self.alphabet),
                kenlm_model_path=f'hw_asr/text_encoder/language_models/{lm_n_gram}-gram.pruned.{lm_prun_thr}.arpa'
            )
        else:
            print('without lm')
            self.decoder = build_ctcdecoder(
                labels=[''] + list(self.alphabet)
            )

    def ctc_decode(self, inds: List[int]) -> str:
        text = []
        prev_char = self.EMPTY_TOK
        for ind in inds:
            if ind == self.char2ind[self.EMPTY_TOK]:
                prev_char = self.ind2char[ind]
                continue
            if self.ind2char[ind] != prev_char:
                text.append(self.ind2char[ind])
            prev_char = self.ind2char[ind]
        text = ''.join(text)
        return text
    
    def _truncate(self, state, beamsize):
        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])
        return dict(state_list[:beamsize])

    def _extend_and_merge(self, frame, state, ind2char):
        new_state = defaultdict(float)
        for next_char_index, next_char_prob in enumerate(frame):
            for (pref, last_char), pref_proba in state.items():
                next_char = ind2char[next_char_index]
                if next_char == last_char:
                    new_pref = pref
                else:
                    if next_char != self.EMPTY_TOK:
                        new_pref = pref + next_char
                    else:
                        new_pref = pref
                    last_char = next_char
                new_state[(new_pref, last_char)] += pref_proba * next_char_prob
        return new_state


    def ctc_beam_search_deprecated(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        :param probs: probability tensor of shape [time_dimension x n_characters]
        :param probs_length: some description
        :param beam_size: beam size
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        state = {('', self.EMPTY_TOK) : 1.0}
        for frame_index in range(probs_length):
            state = self._extend_and_merge(probs[frame_index, :], state, self.ind2char)
            state = self._truncate(state, beam_size)
        hypos = [Hypothesis(text, prob) for (text, _), prob in state.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
    
    def ctc_beam_search(self, probs: torch.tensor, probs_length, beam_size: int):
        text = self.decoder.decode(torch.softmax(probs[:probs_length, :], -1).cpu().detach().numpy(), beam_width=beam_size)
        return [Hypothesis(text, 1)]