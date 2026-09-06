"""Tests for the E5 content probes (scripts/eval_e5.py).

Three things would silently invalidate E5. The targets are read off the drumroll by
hand here, because a wrong step boundary or a max-vs-first mixup makes every probe
number a measurement of the wrong thing while still looking plausible. The per-step
feature must pool the 16 FREQUENCY tokens of a time step and keep the 8 time steps
apart — pooling the wrong axis of the row-major 8x16 grid gives 16 "steps" that each
average all of time, which would destroy the timing signal without any error. And
macro-F1 must match a hand-computed case, since the whole drum-JEPA vs AO-JEPA
comparison is a difference between two of those numbers.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_e5  # noqa: E402

from drumjepa.drum_map import K_V1  # noqa: E402
from drumjepa.features import SEG_FRAMES  # noqa: E402
from tests.test_model import B, make  # noqa: E402

S, P = eval_e5.N_STEPS, eval_e5.STEP_FRAMES  # 8 steps of 25 frames


def test_targets_from_a_hand_made_drumroll():
    """Onsets, max velocity, first-onset frame and mean CC4, on a roll built by hand."""
    a = np.zeros((1, SEG_FRAMES, K_V1), np.float32)
    # Step 0, kick (class 0): two hits, the louder one second. Max = 0.9, first = 3.
    a[0, 3, 0] = 0.4
    a[0, 17, 0] = 0.9
    # Step 2, snare (class 1): one hit at frame 24 of the step (absolute 74).
    a[0, 2 * P + 24, 1] = 0.5
    # Step 7, kick at the step's first frame, to catch an off-by-one at the boundary.
    a[0, 7 * P, 0] = 0.25
    cc = np.tile(np.arange(S, dtype=np.float32)[:, None] / 10.0, (1, P)).reshape(1, SEG_FRAMES)

    t = eval_e5.step_targets(a, cc)
    assert t["onset"].shape == (1, S, K_V1)
    on = np.zeros((S, K_V1), bool)
    on[0, 0] = on[2, 1] = on[7, 0] = True
    assert (t["onset"][0] == on).all(), "an onset is any nonzero cell in the step's 25 frames"
    assert t["velocity"][0, 0, 0] == np.float16(0.9), "velocity is the max, not the first"
    assert t["velocity"][0, 2, 1] == np.float16(0.5)
    assert t["velocity"][0, 7, 0] == np.float16(0.25)
    assert t["timing"][0, 0, 0] == 3, "timing is the FIRST onset's frame within the step"
    assert t["timing"][0, 2, 1] == 24
    assert t["timing"][0, 7, 0] == 0
    assert (t["timing"][0][~on] == -1).all(), "no onset must be flagged, not read as frame 0"
    assert np.allclose(t["hihat"][0].astype(np.float32), np.arange(S) / 10.0, atol=1e-3)


def test_step_features_pool_the_frequency_axis():
    """The pooled feature must vary across the 8 time steps, not across the 16 bands."""
    model, batch = make()
    with torch.no_grad():
        tok = model.encode_state(batch["x_t"], teacher=True)
    n_t, n_f = model.grid
    assert (n_t, n_f) == (8, 16) and tok.shape == (B, n_t * n_f, model.cfg["d_model"])
    z = eval_e5.step_features(tok, model.grid)
    assert z.shape == (B, n_t, model.cfg["d_model"])
    assert torch.allclose(z, tok.reshape(B, n_t, n_f, -1).mean(2), atol=1e-6)
    assert eval_e5.step_tokens(tok, model.grid).shape == (B, n_t, n_f * model.cfg["d_model"])

    # A grid whose time rows differ but whose frequency columns are identical within a row.
    # Pooling over frequency must keep the 8 rows distinct; pooling over time would
    # collapse them to one repeated vector.
    d = 4
    g = torch.arange(n_t, dtype=torch.float32)[:, None, None].expand(n_t, n_f, d)
    grid = g.reshape(1, n_t * n_f, d)
    pooled = eval_e5.step_features(grid, (n_t, n_f))[0]
    assert torch.allclose(pooled[:, 0], torch.arange(n_t, dtype=torch.float32)), \
        "pooled over the wrong axis: the time steps are no longer distinguishable"
    assert len({tuple(r.tolist()) for r in pooled}) == n_t


def test_macro_f1_on_a_hand_made_case():
    """Two classes, counted by hand: F1 = 2/3 and 1.0, macro 5/6."""
    y = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], bool)
    pred = np.array([[1, 1], [0, 0], [1, 1], [0, 0]], bool)
    # class 0: tp 1, fp 1, fn 1 -> 2/(2+1+1) = 0.5. class 1: tp 2, fp 0, fn 0 -> 1.0.
    f1 = eval_e5.f1_from_counts(eval_e5.f1_counts(y, pred))
    assert np.allclose(f1, [0.5, 1.0]), f1
    assert np.isclose(f1.mean(), 0.75)

    seq = np.array([0, 0, 1, 1])
    c = eval_e5.counts_by_seq(y, pred, seq)
    assert c.shape == (2, 2, 3)
    assert np.allclose(eval_e5.f1_from_counts(c.sum(0)), f1), "per-sequence counts must sum back"
    lo, hi = eval_e5.f1_bootstrap(c, n_boot=200, seed=0)
    assert 0.0 <= lo <= f1.mean() <= hi <= 1.0, (lo, hi)

    # A class that is never true and never predicted scores 0, not NaN (sklearn's
    # zero_division=0 convention), so macro-F1 stays a number.
    z = np.zeros((4, 1), bool)
    assert eval_e5.f1_from_counts(eval_e5.f1_counts(z, z)).tolist() == [0.0]


def test_prior_baseline_matches_its_closed_form():
    """The trivial baseline's expected F1 is 2pq/(p+q); check it against a simulation."""
    rng = np.random.default_rng(0)
    p = np.array([0.3, 0.05])
    Y = rng.random((200_000, 2)) < np.array([0.3, 0.05])
    exact = eval_e5.prior_f1(p, Y)
    pred = rng.random(Y.shape) < p
    sim = eval_e5.f1_from_counts(eval_e5.f1_counts(Y, pred))
    assert np.allclose(exact, sim, atol=0.01), (exact, sim)
