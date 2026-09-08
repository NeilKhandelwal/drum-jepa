"""Synthetic-batch tests for drumjepa.model.DrumJEPA (CPU, repeated on MPS)."""
import pytest
import torch

from drumjepa.drum_map import K_V1
from drumjepa.features import N_MELS, SEG_FRAMES
from drumjepa.model import DrumJEPA

B = 4
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def make(device="cpu", cfg=None):
    """Seeded model and synthetic batch on `device`."""
    torch.manual_seed(0)
    model = DrumJEPA(cfg or {}).to(device)
    batch = {
        "x_t": torch.randn(B, SEG_FRAMES, N_MELS, device=device),
        "x_t1": torch.randn(B, SEG_FRAMES, N_MELS, device=device),
        "a_t": (torch.rand(B, SEG_FRAMES, K_V1, device=device) > 0.9).float(),
        "a_t1": (torch.rand(B, SEG_FRAMES, K_V1, device=device) > 0.9).float(),
        "cc_t": torch.rand(B, SEG_FRAMES, device=device),
        "cc_t1": torch.rand(B, SEG_FRAMES, device=device),
        "kit_id": torch.zeros(B, dtype=torch.long),
        "seq_idx": torch.zeros(B, dtype=torch.long),
        "density_bin": torch.zeros(B, dtype=torch.long),
    }
    return model, batch


KEYS = ["loss", "loss_s", "loss_a", "loss_rec", "s_tea_std", "s_stu_std", "a_tea_std", "a_stu_std"]


@pytest.mark.parametrize("device", DEVICES)
def test_shapes_and_finite(device):
    model, batch = make(device)
    assert model.encode_state(batch["x_t"]).shape == (B, 128, 256)
    assert model.encode_action(batch["a_t"], batch["cc_t"]).shape == (B, 8, 256)
    out = model(batch)
    assert set(out) == set(KEYS)
    for k in KEYS:
        assert out[k].ndim == 0 and torch.isfinite(out[k]), k


@pytest.mark.parametrize("device", DEVICES)
def test_teacher_equals_student_at_init(device):
    """The EMA teacher starts as an exact copy, so targets are not arbitrary at step 0."""
    model, batch = make(device)
    with torch.no_grad():
        assert torch.allclose(model.encode_state(batch["x_t"], teacher=True),
                              model.encode_state(batch["x_t"]), atol=1e-6)
        assert torch.allclose(model.encode_action(batch["a_t"], batch["cc_t"], teacher=True),
                              model.encode_action(batch["a_t"], batch["cc_t"]), atol=1e-6)


def test_ema_update_moves_teacher_toward_student():
    """tau controls how fast targets track the student; a wrong rate breaks the target."""
    model, _ = make()
    p_s = model.Es_stu.patch.weight
    p_t = model.Es_tea.patch.weight
    with torch.no_grad():
        p_s.add_(1.0)
    old = p_t.detach().clone()
    model.ema_update()
    assert torch.allclose(p_t, 0.95 * old + 0.05 * p_s, atol=1e-6)


def test_no_grad_into_teachers():
    """Gradient reaching the teacher would be a trivial-solution path (collapse)."""
    model, batch = make()
    model(batch)["loss"].backward()
    for name, p in list(model.Es_tea.named_parameters()) + list(model.Ea_tea.named_parameters()):
        assert p.grad is None, name


def test_grad_into_students_and_predictors():
    model, batch = make()
    model(batch)["loss"].backward()
    for name, mod in (("Es_stu", model.Es_stu), ("Ea_stu", model.Ea_stu),
                      ("f", model.f), ("g", model.g)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), name


def test_visible_only_encoding_hides_masked_tokens():
    """The context encoder must not see masked tokens; if it did, prediction is trivial."""
    model, batch = make()
    model(batch)
    keep = model._visible_idx(model._last_masks["state"])
    assert keep.shape == (B, 32)
    with torch.no_grad():
        part = model.encode_state(batch["x_t1"], keep=keep)
        full = model.encode_state(batch["x_t1"])
    assert part.shape == (B, 32, 256)
    rows = torch.gather(full, 1, keep[..., None].expand(-1, -1, full.size(-1)))
    assert not torch.allclose(part, rows, atol=1e-4)


def test_mask_counts():
    """Exactly round(0.75*N) masked per sample, else the batch is not rectangular."""
    model, batch = make()
    model(batch)
    m = model._last_masks
    assert m["state"].shape == (B, 128) and m["action"].shape == (B, 8)
    assert (m["state"].sum(1) == 96).all()
    assert (m["action"].sum(1) == 6).all()


def test_loss_decreases_on_one_batch():
    """The predictors must be able to fit a fixed batch; a flat loss means a dead path."""
    model, batch = make()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        out = model(batch)
        losses.append(out["loss"].item())
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        model.ema_update()
    assert losses[4] < losses[0], losses


@pytest.mark.skipif("mps" not in DEVICES, reason="no MPS")
def test_mps_bf16_autocast():
    """Training runs under bf16 autocast on MPS: every matmul stays inside the block."""
    model, batch = make("mps")
    with torch.autocast("mps", dtype=torch.bfloat16):
        out = model(batch)
        out["loss"].backward()
    for k in KEYS:
        assert torch.isfinite(out[k]), k
    assert torch.isfinite(model.Es_stu.patch.weight.grad).all()


def test_action_only_predictor_ignores_state():
    """use_state=False (E2 baseline): f's output must not depend on x_t."""
    m, b = make(cfg={"use_state": False})
    m.eval()
    mask = torch.zeros(4, m.n_s, dtype=torch.bool); mask[:, :96] = True
    e1 = m.state_prediction_error(b["x_t"], b["a_t1"], b["cc_t1"], b["x_t1"], mask)
    e2 = m.state_prediction_error(torch.randn_like(b["x_t"]), b["a_t1"], b["cc_t1"], b["x_t1"], mask)
    assert torch.allclose(e1, e2)
    full, _ = make()
    full.eval()
    f1 = full.state_prediction_error(b["x_t"], b["a_t1"], b["cc_t1"], b["x_t1"], mask)
    f2 = full.state_prediction_error(torch.randn_like(b["x_t"]), b["a_t1"], b["cc_t1"], b["x_t1"], mask)
    assert not torch.allclose(f1, f2)


def test_audio_only_ignores_actions():
    """use_action=False (E5 AO-JEPA baseline): loss must not depend on the action."""
    m, b = make(cfg={"use_action": False})
    m.eval()
    mask = torch.zeros(4, m.n_s, dtype=torch.bool); mask[:, :96] = True
    e1 = m.state_prediction_error(b["x_t"], b["a_t1"], b["cc_t1"], b["x_t1"], mask)
    e2 = m.state_prediction_error(b["x_t"], torch.rand_like(b["a_t1"]), b["cc_t1"], b["x_t1"], mask)
    assert torch.allclose(e1, e2)
    out = m(b)
    assert out["loss_a"].item() == 0.0 and torch.isfinite(out["loss"])
    assert "Ea" not in m.n_params() and m.n_params()["total_student"] > 0


def test_aux_reconstruction_head():
    """aux_rec > 0 (option A): loss_rec is finite, trains the state encoder, and is 0 when off."""
    m, b = make(cfg={"aux_rec": 1.0})
    out = m(b)
    assert torch.isfinite(out["loss_rec"]) and out["loss_rec"].item() > 0
    out["loss"].backward()
    assert m.rec_head.weight.grad is not None
    assert any(p.grad is not None for p in m.Es_stu.parameters())
    m0, b0 = make()
    assert m0(b0)["loss_rec"].item() == 0.0 and m0.rec_head is None


def test_aux_mel_target_control():
    """aux_target 'mel' is the control for option A: same head, target carries no action.

    If the mel loss moved with a_t the control would not isolate action content, which
    is the only reason the run exists (docs/followups.md item 2).
    """
    m, b = make(cfg={"aux_rec": 1.0, "aux_target": "mel"})
    assert m.rec_head.out_features == N_MELS
    steps = m.encode_state(b["x_t"]).view(B, m.grid[0], m.grid[1], -1).mean(2)
    assert m.rec_head(steps).shape == (B, 8, N_MELS)
    out = m(b)
    assert torch.isfinite(out["loss_rec"]) and out["loss_rec"].item() > 0
    silent = m({**b, "a_t": torch.zeros_like(b["a_t"])})
    assert silent["loss_rec"].item() == pytest.approx(out["loss_rec"].item())
    out["loss"].backward()
    assert m.rec_head.weight.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.Es_stu.parameters())
