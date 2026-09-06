"""Tests for the E3 kit-latent evaluation (scripts/eval_e3.py).

Three things would silently invalidate E3. The clip embedding must be the mean over
the encoder's tokens and the raw baseline the mean over the clip's frames, or the
"a probe only counts if it beats the raw mel" comparison is between two different
quantities. The geometry statistics must be 1.0 on data where kit really is an
additive offset and ~0 / ~chance on data where it is nothing, or a middling number
on the real embeddings cannot be read either way. And the held-out mapping table is
a distribution over the 6 train kits, so its rows must sum to 1 whatever the probe
predicts.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import eval_e3  # noqa: E402

from tests.test_model import B, make  # noqa: E402

N_KITS, W, D = 6, 60, 32
PAIRS = [(a, b) for a in range(N_KITS) for b in range(N_KITS) if a != b]


def additive_embeddings(seed=0, noise=0.0):
    """(N_KITS, W, D) where kit is an exact additive offset on a shared window content."""
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(W, D))
    offsets = rng.normal(size=(N_KITS, D)) * 6.0
    return content[None] + offsets[:, None] + noise * rng.normal(size=(N_KITS, W, D))


def test_embeddings_are_means_over_tokens_and_over_frames():
    model, batch = make()
    with torch.no_grad():
        z = eval_e3.mean_pool_embed(model, batch["x_t"], teacher=True)
        tokens = model.encode_state(batch["x_t"], teacher=True)
    assert z.shape == (B, model.cfg["d_model"]) and torch.isfinite(z).all()
    assert torch.allclose(z, tokens.float().mean(1), atol=1e-6), "not the mean over tokens"
    raw = eval_e3.raw_mel_embed(batch["x_t"])
    assert raw.shape == (B, batch["x_t"].shape[2]), "raw baseline is one value per mel bin"
    assert torch.allclose(raw, batch["x_t"].float().mean(1), atol=1e-6), "not the mean over frames"


def test_geometry_is_exact_on_additive_kits_and_null_on_random():
    """An exact kit offset must give cosine 1 and transfer 1; noise must give neither."""
    E = additive_embeddings()
    centroids = E.mean(1)
    g = eval_e3.delta_geometry(E, PAIRS, n_boot=20, seed=0)
    assert g["within_pair_cosine"] > 0.999, g
    assert g["within_ci"][0] > 0.99 and g["within_ci"][1] <= 1.0 + 1e-9, g
    # Unrelated directions in D dims average |cos| ~ sqrt(2/(pi*D)); 3x that is generous.
    chance = np.sqrt(2 / (np.pi * D))
    assert g["between_pair_cosine"] < 3 * chance, "disjoint kit pairs share no direction here"
    acc = eval_e3.kit_transfer(E, PAIRS, centroids, seed=0)
    assert min(acc.values()) == 1.0, "an exact offset must land on kit B's centroid every time"

    rng = np.random.default_rng(1)
    R = rng.normal(size=(N_KITS, W, D))
    gr = eval_e3.delta_geometry(R, PAIRS, n_boot=20, seed=0)
    assert abs(gr["within_pair_cosine"]) < 0.1, gr
    assert gr["between_pair_cosine"] < 3 * chance, gr
    ar = eval_e3.kit_transfer(R, PAIRS, R.mean(1), seed=0)
    assert np.mean(list(ar.values())) < 3.0 / N_KITS, "no kit structure, so near chance"


def test_mapping_table_rows_are_distributions():
    rng = np.random.default_rng(0)
    kit_id = np.repeat(np.arange(N_KITS), 40)
    pred = rng.integers(0, 3, len(kit_id))  # a probe that never predicts classes 3..5
    T = eval_e3.mapping_table(pred, kit_id, [3, 4, 5], list(range(N_KITS)))
    assert T.shape == (3, N_KITS)
    assert np.allclose(T.sum(1), 1.0), T.sum(1)
    assert (T[:, 3:] == 0).all(), "unpredicted classes must be zero, not renormalized away"
