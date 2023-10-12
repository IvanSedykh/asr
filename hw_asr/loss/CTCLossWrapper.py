import torch
from torch import Tensor
from torch.nn import CTCLoss


class CTCLossWrapper(CTCLoss):
    def forward(self, log_probs, log_probs_length, text_encoded, text_encoded_length,
                **batch) -> Tensor:
        log_probs_t = torch.transpose(log_probs, 0, 1)

        val = super().forward(
            log_probs=log_probs_t,
            targets=text_encoded,
            input_lengths=log_probs_length,
            target_lengths=text_encoded_length,
        )

        if not torch.isfinite(val):
            # TODO: remove after fixing -- may slow down
            print(f"{log_probs_t.shape=}")
            print(f"{log_probs=}")
            print(f"{text_encoded=}")
            print(f"{log_probs_length=}")
            print(f"{text_encoded_length=}")
            print(f"{torch.any(log_probs_length < text_encoded_length)=}")
            print(f"{(log_probs_length < text_encoded_length)=}")
            print(f"{val=}")
            raise ValueError
        return val
