"""Tests for the fixed-readout prediction metric (scripts/eval_readout.py).

Three things would make every number in this eval wrong while leaving it plausible.
The predicted grid must hold f's tokens at the MASKED positions and the teacher's at
the visible ones: swap them and the metric measures the teacher, which is the ceiling
by construction. The K masked copies of a transition must line up with K copies of
that transition's onset targets, so a tile-vs-repeat mixup does not score each mask
against another mask's targets. And the masked error reported alongside must be the
same quantity E1 and E2 report, or the new metric cannot be compared with the old one.
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_readout  # noqa: E402

from drumjepa.drum_map import K_V1  # noqa: E402
from drumjepa.features import N_MELS, SEG_FRAMES  # noqa: E402
from eval_e1 import masks_for  # noqa: E402
from eval_e5 import N_STEPS, step_features  # noqa: E402
from tests.test_model import make  # noqa: E402

N, K = 3, 2


class FakeDS:
    """A synthetic stand-in for SegmentPairs: N transitions of random tensors."""

    def __init__(self, n=N, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = [{"x_t": torch.randn(SEG_FRAMES, N_MELS, generator=g),
                       "x_t1": torch.randn(SEG_FRAMES, N_MELS, generator=g),
                       "a_t1": (torch.rand(SEG_FRAMES, K_V1, generator=g) > 0.9).float(),
                       "cc_t1": torch.rand(SEG_FRAMES, generator=g),
                       "kit_id": i % 2, "seq_idx": i} for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


ARGS = types.SimpleNamespace(K=K, seed=0, bs=4, workers=0)


def run(model):
    ds = FakeDS()
    feat = eval_readout.run_predictor(ds, np.arange(len(ds)), model,
                                      torch.device("cpu"), False, ARGS)
    return ds, feat


def test_predicted_grid_keeps_the_teacher_at_visible_positions():
    """f's tokens at the masked positions, the teacher's at the rest — not the other way."""
    pred = torch.full((2, 4, 3), 1.0)
    tgt = torch.full((2, 4, 3), -1.0)
    mask = torch.tensor([[True, False, True, False], [False, False, True, True]])
    g = eval_readout.predicted_grid(pred, tgt, mask)
    assert (g[mask] == 1.0).all(), "masked positions must carry f's prediction"
    assert (g[~mask] == -1.0).all(), "visible positions must carry the teacher's token"


def test_grids_and_shapes_on_a_synthetic_batch():
    """Every condition's grid is what its name says, on a real model and fake data."""
    model, _ = make()
    model.eval()
    ds, feat = run(model)
    d = model.cfg["d_model"]
    assert feat["predicted_step"].shape == (N * K, N_STEPS, d)
    assert feat["ceiling_step"].shape == (N, N_STEPS, d)
    assert feat["full_mask_step"].shape == (N, N_STEPS, d)
    assert feat["mask_frac"].shape == (N * K, N_STEPS)
    # 75% of the tokens are masked, so the per-step fractions average to the mask ratio.
    assert np.isclose(feat["mask_frac"].astype(np.float32).mean(), model.mask_ratio, atol=1e-3)

    x_t1 = torch.stack([ds[i]["x_t1"] for i in range(N)])
    with torch.no_grad():
        tea = step_features(model.encode_state(x_t1, teacher=True), model.grid)
    assert np.allclose(feat["ceiling_step"].astype(np.float32), tea.numpy(), atol=2e-3), \
        "the ceiling grid must be the teacher's own tokens of the true x_{t+1}"
    assert not np.allclose(feat["full_mask_step"].astype(np.float32), tea.numpy(), atol=2e-3)


def test_masked_error_matches_the_e1_and_e2_quantity():
    """The reported error must equal model.state_prediction_error under the same masks."""
    model, _ = make()
    model.eval()
    ds, feat = run(model)
    want = []
    with torch.inference_mode():
        for i in range(N):
            it = ds[i]
            mask = masks_for(model, i, K, ARGS.seed, False)
            e = model.state_prediction_error(
                *[t[None].expand(K, *t.shape) for t in
                  (it["x_t"], it["a_t1"], it["cc_t1"], it["x_t1"])], mask)
            want.append(float(e.mean()))
    assert np.allclose(feat["masked_mse"], want, atol=1e-5), (feat["masked_mse"], want)


def test_condition_rows_repeat_targets_once_per_mask():
    """Each of the K masked copies is scored against ITS OWN transition's targets."""
    d = 5
    feat = {"predicted_step": np.arange(N * K * N_STEPS * d, dtype=np.float16).reshape(
                N * K, N_STEPS, d),
            "predicted_clip": np.zeros((N * K, d), np.float16),
            "ceiling_step": np.zeros((N, N_STEPS, d), np.float16),
            "ceiling_clip": np.zeros((N, d), np.float16),
            "onset": np.arange(N)[:, None, None].repeat(N_STEPS, 1).repeat(
                len(eval_readout.CLASSES_V1), 2) > 0,
            "kit_id": np.arange(N), "seq_idx": np.arange(N) * 10}
    r = eval_readout.cond_rows(feat, "predicted", K)
    assert r["step"].shape == (N * K * N_STEPS, d)
    # Transition i owns rows [i*K, (i+1)*K) of the predicted grid, so its targets and
    # sequence id must appear K times in a row, not once every N rows.
    assert r["kit_id"].tolist() == [0, 0, 1, 1, 2, 2]
    assert r["clip_seq"].tolist() == [0, 0, 10, 10, 20, 20]
    assert r["step_seq"].tolist() == sum(([s] * N_STEPS for s in (0, 0, 10, 10, 20, 20)), [])
    assert r["onset"].shape == (N * K * N_STEPS, len(eval_readout.CLASSES_V1))
    assert r["onset"][:K * N_STEPS].sum() == 0, "transition 0 has no onsets in this fixture"

    c = eval_readout.cond_rows(feat, "ceiling", K)
    assert c["step"].shape == (N * N_STEPS, d), "the ceiling grid does not depend on the mask"
    assert c["kit_id"].tolist() == [0, 1, 2]
