"""Tests for the E2 state ablation (scripts/eval_e2.py).

Three things would silently invalidate E2: the ring-out / dense subsets must be read
off the *t+1* window (the dataset's density array is the t window, so a copy-paste
there compares the wrong segments); the action-only baseline must really ignore s_t
(otherwise "kit swap only moves the full model" is not a control); and the paired
aggregation must have the direction of the log ratio right, or the table reads
backwards.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_e2  # noqa: E402

from tests.test_model import B, make  # noqa: E402


def fake_a_t1(counts, T=200, K=14):
    """(len(counts), T, K) drumroll with exactly `counts[i]` nonzero cells in row i."""
    a = torch.zeros(len(counts), T, K)
    for i, c in enumerate(counts):
        flat = a[i].view(-1)
        flat[torch.arange(c) * 7] = 0.5  # velocity/127, any positive value is an onset
    return a


def test_onset_bins_and_subsets_use_the_t1_window():
    counts = [0, 1, 2, 3, 5, 20, 21, 60]
    n, bins = eval_e2.onset_bins(fake_a_t1(counts))
    assert n.tolist() == counts
    assert bins.tolist() == [0, 1, 1, 2, 2, 4, 5, 6]  # DENSITY_EDGES [0,1,3,6,11,21,41]
    sub = eval_e2.subset_masks(bins, n)
    assert sub["ring_out"].tolist() == [c <= 2 for c in counts], "ring_out is density bins 0-1"
    assert sub["dense"].tolist() == [c >= 21 for c in counts], "dense is density bins 5-6"
    assert sub["all"].all()
    # Thin ring_out (< 100 transitions) triggers the <= 5 fallback subset.
    assert sub["ring_out_le5"].tolist() == [c <= 5 for c in counts]
    assert "ring_out_le5" not in eval_e2.subset_masks(np.zeros(200, np.int64), np.zeros(200, np.int64))


def test_action_only_is_kit_swap_invariant_and_full_model_is_not():
    """The baseline's advantage under kit swap must be exactly zero by construction."""
    full, batch = make()
    action_only, _ = make(cfg={"use_state": False})
    mask = full._state_mask(B, torch.device("cpu"))
    x_swap = torch.randn_like(batch["x_t"])  # same window on another kit, stand-in
    for model, invariant in ((action_only, True), (full, False)):
        args = (batch["a_t1"], batch["cc_t1"], batch["x_t1"], mask)
        clean = model.state_prediction_error(batch["x_t"], *args)
        swap = model.state_prediction_error(x_swap, *args)
        d = (clean - swap).abs().max().item()
        assert (d < 1e-3) == invariant, f"invariant={invariant} but max |clean - swap| = {d:.3e}"


def test_aggregate_win_rate_and_log_ratio_direction():
    """Positive log ratio and win rate > 0.5 must both mean 'the full model is better'."""
    e_full = np.array([1.0, 1.0, 1.0, 2.0])
    e_ao = np.array([np.e, np.e, np.e, 1.0])  # full wins 3 of 4; log ratios 1,1,1,log(0.5)
    seq = np.array([0, 1, 2, 3])
    r = eval_e2.aggregate(e_full, e_ao, seq, seed=0)
    assert r["n"] == 4 and r["win_rate"] == 0.75
    assert r["err_full"] == 1.25 and abs(r["err_action_only"] - (3 * np.e + 1) / 4) < 1e-12
    assert abs(r["log_ratio"] - (3 - np.log(2)) / 4) < 1e-12
    for lo, hi, v in ((*r["win_ci"], r["win_rate"]), (*r["log_ratio_ci"], r["log_ratio"])):
        assert lo <= v <= hi, (lo, v, hi)
    # Swapping the models flips both statistics.
    s = eval_e2.aggregate(e_ao, e_full, seq, seed=0)
    assert s["win_rate"] == 0.25 and abs(s["log_ratio"] + r["log_ratio"]) < 1e-12
