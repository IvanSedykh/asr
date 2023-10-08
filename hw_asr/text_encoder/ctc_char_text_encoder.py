from typing import List, NamedTuple
from collections import defaultdict

import numpy as np
import torch
import pyctcdecode

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    log_prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = ""  # to use pyctcdecode empty token has to be ""
    EMPTY_TOK_ID = 0

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.decoder = pyctcdecode.build_ctcdecoder(vocab)

    def ctc_decode(self, inds: List[int]) -> str:
        # TODO: your code here
        output_ids = []
        prev = self.EMPTY_TOK_ID
        for token_id in inds:
            if token_id != prev:
                if token_id != self.EMPTY_TOK_ID:
                    output_ids.append(token_id)
            prev = token_id
        output_text = self.decode(output_ids)
        return output_text

    def _extend_and_merge(
        self, hypos: List[Hypothesis], probs_frame: torch.tensor, beam_size: int = 100
    ) -> List[Hypothesis]:
        assert len(probs_frame.shape) == 1
        new_hypos = defaultdict([])  # prefix -> list[log_prob]
        # take top probs to reduce compute
        top_probs, top_inds = torch.topk(
            probs_frame, k=min(beam_size, len(self.ind2char))
        )
        top_inds = top_inds.tolist()
        for next_token_id, next_token_prob in zip(top_inds, top_probs):
            next_token = self.ind2char[next_token_id]
            for hypo in hypos:
                prefix = hypo.text
                last_token = prefix[-1] if len(prefix) > 0 else self.EMPTY_TOK
                if next_token != last_token:
                    prefix = prefix + next_token
                    last_token = next_token
                # save probs for same prefixes
                new_hypos[prefix].append(hypo.log_prob + next_token_prob)
        return [
            Hypothesis(text=k, log_prob=torch.logsumexp(torch.tensor(v), 0).item())
            for k, v in new_hypos.items()
        ]

    def _truncate(
        self, hypos: List[Hypothesis], beam_size: int = 100
    ) -> List[Hypothesis]:
        return sorted(hypos, key=lambda x: x.log_prob, reverse=True)[:beam_size]

    def _remove_empty_token(self, text: str):
        return text.replace(self.EMPTY_TOK, "")


    # this is working implementation of ctc beam search
    def ctc_beam_search_my(
        self, log_probs: torch.tensor, probs_length, beam_size: int = 100
    ) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = []

        hypos.append(Hypothesis(text="", log_prob=0.0))
        log_probs = log_probs[:probs_length]  # remove padding

        for frame_num, probs_frame in enumerate(log_probs):
            hypos = self._extend_and_merge(hypos, probs_frame, beam_size)
            hypos = self._truncate(hypos, beam_size)

        # remove empty tokens
        hypos = [
            Hypothesis(text=self._remove_empty_token(hypo.text), log_prob=hypo.log_prob)
            for hypo in hypos
        ]

        return sorted(hypos, key=lambda x: x.log_prob, reverse=True)


    # but i will use the implementation from pyctcdecode
    # because it is faster and more stable
    def ctc_beam_search(
        self, log_probs: torch.tensor, probs_length, beam_size: int = 100
    ) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)

        log_probs = log_probs[:probs_length]  # remove padding
        log_probs = np.array(log_probs.detach())

        beam_outs = self.decoder.decode_beams(
            logits=np.exp(log_probs), beam_width=beam_size
        )
        hypos = []
        for beam_out in beam_outs:
            hypos.append(
                Hypothesis(text=beam_out[0], log_prob=beam_out[3])
            )
        return hypos
