"""Tests for the E1 evaluation (scripts/eval_e1.py) and its model hook.

E1 is the gate for every later experiment, so the two things that can silently
invalidate it are pinned here: the per-sample error must be the same quantity the
model trains on (else the win rates measure something else), and the perturbation
lookups must pick the intended neighbour / kit counterfactual (else "kit swap"
compares a transition to itself).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_e1  # noqa: E402

from tests.test_model import B, make  # noqa: E402


def test_state_prediction_error_matches_loss_s():
    """Per-sample error must average to the trained loss_s under the same mask."""
    model, batch = make()
    fixed = model._state_mask(B, torch.device("cpu"))
    model._state_mask = lambda b, device: fixed
    loss_s = model(batch)["loss_s"]
    err = model.state_prediction_error(batch["x_t"], batch["a_t1"], batch["cc_t1"],
                                       batch["x_t1"], fixed)
    assert err.shape == (B,) and torch.isfinite(err).all()
    assert torch.allclose(err.mean(), loss_s, atol=1e-4), (err.mean(), loss_s)


def test_full_mask_has_no_visible_tokens():
    """--full-mask leaves zero visible t+1 tokens; the predictor must still run."""
    model, batch = make()
    full = torch.ones(B, model.n_s, dtype=torch.bool)
    err = model.state_prediction_error(batch["x_t"], batch["a_t1"], batch["cc_t1"],
                                       batch["x_t1"], full)
    assert err.shape == (B,) and torch.isfinite(err).all()


class FakeDS:
    """Index arrays only: build_sources reads seq_idx / kit_id / offset and len()."""

    def __init__(self, n_kits=3, n_seqs=2, offsets=(0, 200, 400)):
        rows = [(s, k, o) for s in range(n_seqs) for k in range(n_kits) for o in offsets]
        self.seq_idx, self.kit_id, self.offset = (np.array(c) for c in zip(*rows))

    def __len__(self):
        return len(self.offset)


def test_build_sources_picks_neighbour_and_other_kit():
    ds = FakeDS()
    src = eval_e1.build_sources(ds, 200, np.random.default_rng(0))
    for i in range(len(ds)):
        s, k, o = ds.seq_idx[i], ds.kit_id[i], ds.offset[i]
        j = src["state_shift"][i]
        # offset 0 has no earlier window and is skipped (-1): the later window's x_t
        # would be this transition's own x_{t+1}. The rest go back one hop.
        if o == 0:
            assert j == -1
        else:
            assert ds.offset[j] == o - 200
            assert ds.seq_idx[j] == s and ds.kit_id[j] == k, "shift must stay on the same (seq, kit)"
        j = src["kit_swap"][i]
        assert ds.seq_idx[j] == s and ds.offset[j] == o, "kit swap must hold the window fixed"
        assert ds.kit_id[j] != k, "kit swap must change the kit"
        assert ds.seq_idx[src["state_random"][i]] != s
        assert ds.seq_idx[src["action_random"][i]] != s
        assert src["action_shift"][i] == i
    assert ((src["state_shift"] >= 0) == (ds.offset > 0)).all() and (src["kit_swap"] >= 0).all()


def test_masks_are_paired_across_conditions():
    """Clean and perturbed conditions must see identical masks, or the pairing is broken."""
    model, _ = make()
    a = eval_e1.masks_for(model, 7, 4, 0, False)
    assert a.shape == (4, model.n_s) and (a.sum(1) == model.n_mask_s).all()
    assert torch.equal(a, eval_e1.masks_for(model, 7, 4, 0, False))
    assert not torch.equal(a, eval_e1.masks_for(model, 8, 4, 0, False))
