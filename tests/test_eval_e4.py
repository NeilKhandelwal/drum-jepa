"""Tests for the E4 inverse model (drumjepa/inverse.py) and its scoring (scripts/eval_e4.py).

Three things would silently invalidate E4. h must actually be trainable from the two
state embeddings -- shapes right, loss finite, gradient reaching every input path --
or a low F1 says nothing about the representation. The onset decoder and matcher must
count a hit inside the 50 ms tolerance and a miss outside it, and must not turn one
smeared onset into several detections, or precision is measured against a decoder bug
rather than against the model. And the raw-mel baseline must patch the spectrogram
exactly the way the encoder's patch-embed does, or "the trained encoder beats raw mel"
compares two different inputs.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_e4  # noqa: E402

from drumjepa.drum_map import K_V1  # noqa: E402
from drumjepa.features import N_MELS, SEG_FRAMES  # noqa: E402
from drumjepa.inverse import InverseModel, inverse_loss  # noqa: E402
from drumjepa.model import DrumJEPA  # noqa: E402

B, N_TOK, D = 4, 128, 256
POS_WEIGHT = torch.tensor(50.0)


def synth(n=B, seed=0, d_in=D):
    """Frozen-looking embeddings and a sparse action, on CPU."""
    g = torch.Generator().manual_seed(seed)
    return {"z_t": torch.randn(n, N_TOK, d_in, generator=g),
            "z_t1": torch.randn(n, N_TOK, d_in, generator=g),
            "a_t": (torch.rand(n, SEG_FRAMES, K_V1, generator=g) > 0.98).float(),
            "a_t1": (torch.rand(n, SEG_FRAMES, K_V1, generator=g) > 0.98).float(),
            "cc_t": torch.rand(n, SEG_FRAMES, generator=g),
            "cc_t1": torch.rand(n, SEG_FRAMES, generator=g)}


@pytest.mark.parametrize("use_at", [False, True])
def test_forward_shapes_loss_and_gradients(use_at):
    """h must decode the full 200x14 roll and 200-frame CC4, and be trainable."""
    torch.manual_seed(0)
    b = synth()
    model = InverseModel(use_at=use_at)
    at = (b["a_t"], b["cc_t"]) if use_at else (None, None)
    z_t = b["z_t"].requires_grad_(True)
    out = model(z_t, b["z_t1"], *at)
    assert out["onset_logit"].shape == (B, SEG_FRAMES, K_V1)
    assert out["velocity"].shape == (B, SEG_FRAMES, K_V1)
    assert out["cc"].shape == (B, SEG_FRAMES)

    ls = inverse_loss(out, b["a_t1"], b["cc_t1"], POS_WEIGHT)
    assert all(torch.isfinite(v) for v in ls.values()), ls
    ls["loss"].backward()
    assert torch.isfinite(z_t.grad).all() and z_t.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # The a_t path only exists in the with_at variant, and must be used when it does.
    assert hasattr(model, "act_proj") == use_at
    if use_at:
        assert model.act_proj.weight.grad.abs().sum() > 0


def test_with_at_changes_the_prediction():
    """Reading a_t must matter: identical states with different a_t must not tie."""
    torch.manual_seed(0)
    b = synth()
    model = InverseModel(use_at=True)
    o1 = model(b["z_t"], b["z_t1"], b["a_t"], b["cc_t"])
    o2 = model(b["z_t"], b["z_t1"], torch.zeros_like(b["a_t"]), torch.zeros_like(b["cc_t"]))
    assert not torch.allclose(o1["onset_logit"], o2["onset_logit"])


def test_decode_onsets_nms_collapses_a_plateau():
    """A 3-frame plateau above threshold is one onset, at its first frame."""
    prob = np.zeros((1, 20, 1), np.float32)
    prob[0, 10:13, 0] = 0.9
    keep = eval_e4.decode_onsets(prob)
    assert np.flatnonzero(keep[0, :, 0]).tolist() == [10]

    # Two separate peaks further apart than the NMS window stay two onsets.
    prob = np.zeros((1, 20, 1), np.float32)
    prob[0, 4, 0] = prob[0, 12, 0] = 0.9
    assert np.flatnonzero(eval_e4.decode_onsets(prob)[0, :, 0]).tolist() == [4, 12]

    # Below threshold is not an onset however peaked it is.
    assert not eval_e4.decode_onsets(np.full((1, 20, 1), 0.4, np.float32)).any()


def test_match_onsets_tolerance():
    """+3 frames is inside the 50 ms tolerance, +7 is outside it."""
    assert eval_e4.match_onsets(np.array([50]), np.array([53])) == [(50, 53)]
    assert eval_e4.match_onsets(np.array([50]), np.array([57])) == []
    # One prediction cannot satisfy two true onsets: greedy in time order, 50 takes it.
    pairs = eval_e4.match_onsets(np.array([50, 53]), np.array([54]))
    assert pairs == [(50, 54)] and len(pairs) == 1


def test_count_batch_tp_fp_fn_and_velocity_error():
    """The (tp, fp, fn) bookkeeping and the velocity error read the matched cells."""
    roll = np.zeros((1, 200, K_V1), np.float32)
    roll[0, 50, 0] = 100 / 127.0          # matched at +3
    roll[0, 100, 1] = 80 / 127.0          # missed: prediction 7 frames late
    prob = np.zeros((1, 200, K_V1), np.float32)
    prob[0, 53, 0] = 0.9
    prob[0, 107, 1] = 0.9
    prob[0, 20, 2] = 0.9                  # a class with no true onset: pure false positive
    vel = np.zeros((1, 200, K_V1), np.float32)
    vel[0, 53, 0] = 90 / 127.0
    tp, fp, fn, verr, vn = eval_e4.count_batch(prob, roll, vel)
    assert tp[0, 0].sum() == 1 and fp[0, 0].sum() == 2 and fn[0, 0].sum() == 1
    assert tp[0, 0, 0] == 1 and fn[0, 0, 1] == 1 and fp[0, 0, 2] == 1
    assert vn[0, 0] == 1 and 127 * verr[0, 0] == pytest.approx(10.0, abs=1e-3)


def test_threshold_sweep_matches_decoding_at_each_threshold():
    """Sweeping thresholds must give exactly what decoding at each threshold gives.

    The sweep reuses one set of NMS peaks for every threshold; if that shortcut ever
    diverged from decode_onsets, the tuned threshold would be picked on numbers the
    reported metric does not reproduce.
    """
    rng = np.random.default_rng(0)
    prob = rng.random((6, 200, K_V1)).astype(np.float32) ** 3
    roll = (rng.random((6, 200, K_V1)) > 0.99).astype(np.float32)
    vel = rng.random((6, 200, K_V1)).astype(np.float32)
    ths = (0.1, 0.35, 0.5, 0.8)
    swept = eval_e4.count_batch(prob, roll, vel, ths)
    for ti, th in enumerate(ths):
        # A single-threshold call is the reference; its peaks are gated at that threshold.
        one = eval_e4.count_batch(prob, roll, vel, (th,))
        for a, b in zip(swept, one):
            assert np.array_equal(a[ti], b[0]), th
        # And the predicted-onset count must equal decode_onsets at the same threshold.
        assert (swept[0][ti] + swept[1][ti]).sum() == eval_e4.decode_onsets(prob, th).sum()


def test_raw_mel_tokens_match_the_encoder_patch_embed():
    """The raw baseline must be the encoder's patches, or the comparison is unfair."""
    torch.manual_seed(0)
    enc = DrumJEPA({}).Es_stu
    x = torch.randn(2, SEG_FRAMES, N_MELS)
    tokens = eval_e4.raw_mel_tokens(x)
    assert tokens.shape == (2, enc.grid[0] * enc.grid[1], 25 * 15)
    # A Conv2d over the padded mel with the encoder's kernel is a per-patch dot product,
    # so folding the same weights over our tokens must reproduce it exactly.
    padded = torch.nn.functional.pad(x, (0, enc.n_mels_pad - x.size(-1)))
    ref = enc.patch(padded[:, None]).flatten(2).transpose(1, 2)
    w = enc.patch.weight.reshape(enc.patch.weight.size(0), -1)
    assert torch.allclose(tokens @ w.T + enc.patch.bias, ref, atol=1e-4)


def test_overfits_a_handful_of_transitions():
    """200 steps on 16 fixed transitions must cut the onset loss by more than half."""
    torch.manual_seed(0)
    b = synth(n=16, seed=1)
    model = InverseModel()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    first = last = None
    for step in range(200):
        ls = inverse_loss(model(b["z_t"], b["z_t1"]), b["a_t1"], b["cc_t1"], POS_WEIGHT)
        opt.zero_grad(set_to_none=True)
        ls["loss"].backward()
        opt.step()
        first = float(ls["bce"]) if step == 0 else first
        last = float(ls["bce"])
    assert last < 0.5 * first, (first, last)
