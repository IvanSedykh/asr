import math
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding

from hw_asr.base import BaseModel


# from lucidrains
def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class PointWiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor):
        return self.conv(x)


class DepthWiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding="same", groups=in_channels
        )

    def forward(self, x: Tensor):
        return self.conv(x)


class Transpose(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor):
        return x.transpose(*self.dims)


class Swish(nn.Module):
    def forward(self, x: Tensor):
        return x * torch.sigmoid(x)


class ConvModule(nn.Module):
    """Conformer ConvModule
    - input of shape [bs, seq_len, dim]
    - output of shape [bs, seq_len, dim]
    """

    expansion_factor = 2

    def __init__(self, dim: int, kernel_size: int = 31, dropout_p=0.0):
        # we use 31 as kernel size because it is the largest odd number that
        # is less than 32 used in paper (torch doesnt like even kernel sizes)
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose(1, 2),  # [bs, dim, seq_len]
            PointWiseConv(dim, dim * 2 * self.expansion_factor),
            nn.GLU(dim=1),
            DepthWiseConv(
                dim * self.expansion_factor, dim * self.expansion_factor, kernel_size
            ),
            nn.BatchNorm1d(dim * self.expansion_factor),
            Swish(),
            PointWiseConv(dim * self.expansion_factor, dim),
            nn.Dropout(dropout_p),
            Transpose(1, 2),  # [bs, seq_len, dim]
        )

    def forward(self, x: Tensor):
        # residual connection
        return self.block(x) + x


class FeedForwardModule(nn.Module):
    """Conformer FeedForward
    - input of shape [bs, seq_len, dim]
    - output of shape [bs, seq_len, dim]
    """

    def __init__(self, dim, hidden_dim=None, dropout_prob=0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            Swish(),  # todo: try another
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        # idk why the fuck we need this multiplier here
        return x + 0.5 * self.block(x)


# heavily inspired by opensource conformer
# modified to use RoPE instead of relative attn -- we live in 2023
class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with RoPE.
    This concept was proposed in the "RoFormer"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
        learned_freq_rope=False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # todo fix learned_freq=True
        self.rotary_emb = RotaryEmbedding(
            dim=self.d_head, learned_freq=learned_freq_rope
        )

        self.dropout = nn.Dropout(p=dropout_p)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = (
            self.key_proj(key)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.value_proj(value)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )

        # apply rope
        query = self.rotary_emb.rotate_queries_or_keys(query)
        key = self.rotary_emb.rotate_queries_or_keys(key)

        content_score = torch.matmul((query).transpose(1, 2), key.transpose(2, 3))
        score = content_score / self.sqrt_dim

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context)


class MHSAModule(nn.Module):
    def __init__(self, dim, num_heads, dropout_prob=0.0, learned_freq_rope=False):
        super().__init__()

        # todo: add positional embeddings
        self.layernorm = nn.LayerNorm(dim)
        # self.attn = nn.MultiheadAttention(
        #     dim, num_heads, dropout_prob, batch_first=True, need_weights=False
        # )
        self.attn = RelativeMultiHeadAttention(
            d_model=dim,
            num_heads=num_heads,
            dropout_p=dropout_prob,
            learned_freq_rope=learned_freq_rope,
        )
        self.dropout = nn.Dropout(dropout_prob)

    def _residual(self, x):
        x = self.layernorm(x)
        x = self.attn(x, x, x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        return x + self._residual(x)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        dropout_p=0.0,
        num_heads=4,
        kernel_size=31,
        learned_freq_rope=False,
    ):
        super().__init__()
        self.block = nn.Sequential(
            FeedForwardModule(dim, hidden_dim=hidden_dim, dropout_prob=dropout_p),
            MHSAModule(
                dim,
                num_heads=num_heads,
                dropout_prob=dropout_p,
                learned_freq_rope=learned_freq_rope,
            ),
            ConvModule(dim, kernel_size=kernel_size, dropout_p=dropout_p),
            FeedForwardModule(dim, hidden_dim=hidden_dim, dropout_prob=dropout_p),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.block(x)


class ConvSubsampling(nn.Module):
    # inspired by some github code
    def __init__(self, n_feats_in: int, dim: int) -> None:
        """
        assuming it will decrease seq_len by 4 times

        - input of shape [bs, seq_len, n_feats]
        - output of shape [bs, new_seq_len, dim]
        """
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, stride=2),
            Swish(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2),
            Swish(),
        )

        self.proj = nn.Linear(dim * (((n_feats_in - 1) // 2 - 1) // 2), dim)
        # self.proj = nn.Linear(dim * (((n_feats_in - 1) // 2)), dim)

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # add channel dim
        # [bs, 1, seq_len, n_feats]
        x = self.convblock(x)
        # [bs, dim, new_seq_len, new_dim]
        x = x.permute(0, 2, 1, 3)
        # [bs, new_seq_len, dim, new_dim]
        bs, new_seq_len, dim, new_dim = x.shape
        x = x.contiguous().reshape(bs, new_seq_len, dim * new_dim)
        # [bs, new_seq_len, new_dim]
        out = self.proj(x)
        # [bs, new_seq_len, dim]
        return out

    def transform_input_lengths(self, input_lengths):
        def one_conv(input, kernel=3, stride=2, padding=0):
            return (input - kernel + 2 * padding) // stride + 1

        # for two conv layers
        return one_conv(one_conv(input_lengths))
        # return (input_lengths >> 2) - 1

        # for one conv layer
        # return one_conv(input_lengths)


class Conformer(BaseModel):
    def __init__(
        self,
        n_feats,
        n_class,
        dim,
        hidden_dim=None,
        dropout_p=0.0,
        num_heads=4,
        kernel_size=31,
        num_blocks=8,
        do_conv_subsampling=False,
        learned_freq_rope=False,
        **batch
    ):
        super().__init__(n_feats, n_class, **batch)

        self.do_conv_subsampling = do_conv_subsampling
        if do_conv_subsampling:
            self.conv_subsampling = ConvSubsampling(n_feats_in=n_feats, dim=dim)
        else:
            self.conv_subsampling = nn.Linear(n_feats, dim)

        self.blocks = nn.Sequential(
            *[
                ConformerBlock(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    dropout_p=dropout_p,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    learned_freq_rope=learned_freq_rope,
                )
                for _ in range(num_blocks)
            ]
        )
        self.clf = nn.Linear(dim, n_class)

    def forward(self, spectrogram, **batch):
        # [bs, n_feats, seq_len] -> [bs, seq_len, n_feats]
        spectrogram = torch.transpose(spectrogram, 1, 2)

        x = self.conv_subsampling(spectrogram)

        x = self.blocks(x)
        logits = self.clf(x)

        output = {"logits": logits}
        return output

    def transform_input_lengths(self, input_lengths):
        if self.do_conv_subsampling:
            return self.conv_subsampling.transform_input_lengths(input_lengths)
        else:
            return input_lengths
