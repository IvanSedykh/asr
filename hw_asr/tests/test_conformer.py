import pytest
import torch

from hw_asr.model.conformer import (
    ConvModule,
    FeedForwardModule,
    MHSAModule,
    ConformerBlock,
    ConvSubsampling,
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


@pytest.mark.parametrize("dim", [124, 144, 80])
@pytest.mark.parametrize("seq_len", [100, 111, 143])
def test_conv_subsampling(seq_len, dim):
    bs = 4
    # seq_len = 100
    # dim = 128
    spec = torch.randn(bs, seq_len, dim)

    enc_dim = 144
    conv_subs = ConvSubsampling(n_feats_in=dim, dim=enc_dim)

    out = conv_subs(spec)

    def one_conv(input, kernel=3, stride=2, padding=0):
        return (input - kernel + 2 * padding) // stride + 1

    new_seq_len = one_conv(one_conv(seq_len))
    assert out.shape == (spec.shape[0], new_seq_len, enc_dim)


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


def test_conformer_subsampling():
    bs = 4
    seq_len = 100
    dim = 128
    spec = torch.randn(bs, seq_len, dim)

    # idk why the fuck the input is this shape
    inputs = {"spectrogram": spec.transpose(1, 2)}

    n_letters = 26
    net = Conformer(n_feats=dim, n_class=n_letters, dim=dim // 2, conv_subsampling=True)

    output = net(**inputs)

    assert isinstance(output, dict)
    assert "logits" in output

    input_lenght = torch.tensor([seq_len] * bs)
    output_lenght = net.transform_input_lengths(input_lenght)
    assert output["logits"].shape == (bs, output_lenght[0].item(), n_letters)
