import pytest
import torch

from hw_asr.model.conformer import (
    ConvModule,
    FeedForwardModule,
    MHSAModule,
    ConformerBlock,
    Conformer,
)


@pytest.fixture
def spec():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)
    return spec


def test_convmodule():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)
    conv = ConvModule(dim)

    out = conv(spec)
    assert out.shape == spec.shape


def test_feedforward():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)
    ff = FeedForwardModule(dim)

    out = ff(spec)
    assert out.shape == spec.shape


def test_mhsa():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)
    mhsa = MHSAModule(dim, num_heads=4)

    out = mhsa(spec)
    assert out.shape == spec.shape


def test_conformer_block():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)

    confblock = ConformerBlock(dim)

    out = confblock(spec)

    assert out.shape == spec.shape


def test_conformer():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)

    # idk why the fuck the input is this shape
    inputs = {"spectrogram": spec.transpose(1, 2)}

    n_letters = 26
    net = Conformer(n_feats=dim, n_class=n_letters, dim=dim // 2)

    output = net(**inputs)

    assert isinstance(output, dict)
    assert "logits" in output
    assert output["logits"].shape == (bs, seq_len, n_letters)
