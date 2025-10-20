import os

import torch
import torchac


def estimate_bitrate_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
    bitrate = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
    return bitrate


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.0)
    return cdf_with_0


def save_byte_stream(prob, sym, save_name):
    # torch.Size([178724, 256]) torch.Size([178724])
    bt, Q_len = prob.shape
    prob = prob.view(bt, Q_len)
    sym = sym.view(sym.shape[0])

    # Convert to a torchac-compatible CDF.
    output_cdf = pmf_to_cdf(prob)

    # torchac expects sym as int16, see README for details.
    sym = sym.to(torch.int16)

    # torchac expects CDF and sym on CPU.
    output_cdf = output_cdf.detach().cpu()
    sym = sym.detach().cpu()

    # Get real bitrate from the byte_stream.
    byte_stream = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    real_bits = len(byte_stream) * 8

    if not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name))

    with open(save_name, "wb") as fout:
        fout.write(byte_stream)

    # Read from a file.
    with open(save_name, "rb") as fin:
        byte_stream = fin.read()
    assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)

    return real_bits


def get_symbol_from_byte_stream(byte_stream, prob):
    bt, Q_len = prob.shape
    prob = prob.view(bt, Q_len)

    output_cdf = pmf_to_cdf(prob).detach().cpu()
    return torchac.decode_float_cdf(output_cdf, byte_stream)
