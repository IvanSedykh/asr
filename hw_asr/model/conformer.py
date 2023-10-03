import torch
from torch import nn, Tensor

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


class MHSAModule(nn.Module):
    def __init__(self, dim, num_heads, dropout_prob=0.0):
        super().__init__()

        # todo: add positional embeddings
        self.layernorm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout_prob, batch_first=True
        )
        self.dropout = nn.Dropout(dropout_prob)

    def _residual(self, x):
        x = self.layernorm(x)
        x, attn_weights = self.attn(x, x, x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        return x + self._residual(x)


class ConformerBlock(nn.Module):
    def __init__(
        self, dim, hidden_dim=None, dropout_p=0.0, num_heads=4, kernel_size=31
    ):
        super().__init__()
        self.block = nn.Sequential(
            FeedForwardModule(dim, hidden_dim=hidden_dim, dropout_prob=dropout_p),
            MHSAModule(dim, num_heads=num_heads, dropout_prob=dropout_p),
            ConvModule(dim, kernel_size=kernel_size, dropout_p=dropout_p),
            FeedForwardModule(dim, hidden_dim=hidden_dim, dropout_prob=dropout_p),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return self.block(x)


# todo: conv subsampling


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
        **batch
    ):
        super().__init__(n_feats, n_class, **batch)

        self.proj = nn.Linear(n_feats, dim)
        # todo: add conv subsampling
        self.blocks = nn.Sequential(
            *[
                ConformerBlock(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    dropout_p=dropout_p,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.clf = nn.Linear(dim, n_class)

    def forward(self, spectrogram, **batch):
        # [bs, n_feats, seq_len] -> [bs, seq_len, n_feats]
        spectrogram = torch.transpose(spectrogram, 1, 2)

        # todo: conv subsampling
        x = self.proj(spectrogram)

        x = self.blocks(x)
        logits = self.clf(x)

        output = {"logits": logits}
        return output

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here