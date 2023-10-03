from typing import Union
import torch
from torch import Tensor
import torch.nn as nn


from hw_asr.base import BaseModel

class BasicTransformer(BaseModel):
    def __init__(self, n_feats, n_class, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.down_projection = nn.Linear(n_feats, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True, dim_feedforward=128)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(64, n_class)


    def forward(self, spectrogram:Tensor, **batch) -> Tensor | dict:
        # [bs, n_feats, seq_len] -> [bs, seq_len, n_feats]
        spectrogram = spectrogram.transpose(1, 2)

        # [bs, seq_len, n_feats] -> [bs, seq_len, 64]
        spectrogram = self.down_projection(spectrogram)

        output = self.transformer_encoder(spectrogram)
        logits = self.fc(output)
        return {"logits": logits}
    
    def transform_input_lengths(self, input_lengths):
        return input_lengths