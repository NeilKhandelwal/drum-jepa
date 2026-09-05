"""Action-conditioned drum JEPA (Music-JEPA architecture, Algorithm 1 naming).

State encoder Es and action encoder Ea each have an EMA teacher copy. The state
predictor f maps (student s_t tokens, masked student s_{t+1} tokens) to the
teacher s_{t+1} tokens, cross-attending to student a_{t+1} tokens at every layer.
The action predictor g is the same block without cross-attention.

Shapes for the default config: state 200x229 log-mel -> 8x16 = 128 tokens,
action 200x14 drumroll + CC4 -> 8 tokens, d_model 256.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from drumjepa.drum_map import K_V1
from drumjepa.features import N_MELS, SEG_FRAMES

DEFAULTS = dict(
    d_model=256, d_ff=512, n_heads=4,
    enc_s_layers=12, enc_a_layers=8, pred_layers=6,
    state_patch=(25, 15), n_mels_pad=240, action_patch_frames=25,
    mask_ratio=0.75, mask_blocks=4, mask_aspect=(0.75, 1.5), mask_scale=(0.15, 0.2),
    tau=0.95, lambda_a=0.5, dropout=0.0,
)


def sincos_1d(dim, pos):
    """Fixed 1-D sin-cos embedding for positions `pos` (M,) -> (M, dim)."""
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
    omega = 1.0 / 10000.0 ** omega
    out = pos[:, None].float() * omega[None, :]
    return torch.cat([out.sin(), out.cos()], dim=1)


def sincos_2d(dim, h, w):
    """Fixed 2-D sin-cos embedding on an h x w grid, row-major -> (h*w, dim)."""
    rows = torch.arange(h).repeat_interleave(w)
    cols = torch.arange(w).repeat(h)
    return torch.cat([sincos_1d(dim // 2, rows), sincos_1d(dim // 2, cols)], dim=1)


def gather_tokens(x, idx):
    """Select tokens (B, N, d) -> (B, n_keep, d) by index (B, n_keep); None is a no-op."""
    if idx is None:
        return x
    return torch.gather(x, 1, idx[..., None].expand(-1, -1, x.size(-1)))


class MHA(nn.Module):
    """Multi-head attention over `x`, or cross-attention with K/V from `ctx`."""

    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, ctx=None):
        ctx = x if ctx is None else ctx
        B, N, D = x.shape
        h = self.n_heads
        q = self.q(x).view(B, N, h, D // h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.size(1), 2, h, D // h).permute(2, 0, 3, 1, 4)
        p = self.dropout if self.training else 0.0
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """Pre-norm transformer block; with `cross`, a K/V-only cross-attention layer."""

    def __init__(self, d, n_heads, d_ff, cross=False, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = MHA(d, n_heads, dropout)
        self.cross = cross
        if cross:
            self.norm_q = nn.LayerNorm(d)
            self.norm_kv = nn.LayerNorm(d)
            self.xattn = MHA(d, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_ff, d), nn.Dropout(dropout))

    def forward(self, x, ctx=None):
        x = x + self.attn(self.norm1(x))
        if self.cross:
            x = x + self.xattn(self.norm_q(x), self.norm_kv(ctx))
        return x + self.mlp(self.norm2(x))


class StateEncoder(nn.Module):
    """Log-mel (B, T, N_MELS) -> (B, n_t*n_f, d) post-LayerNorm tokens."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        pt, pf = cfg["state_patch"]
        self.n_mels_pad = cfg["n_mels_pad"]
        self.grid = (SEG_FRAMES // pt, self.n_mels_pad // pf)
        self.patch = nn.Conv2d(1, d, kernel_size=(pt, pf), stride=(pt, pf))
        self.register_buffer("pos", sincos_2d(d, *self.grid)[None], persistent=False)
        self.blocks = nn.ModuleList(
            Block(d, cfg["n_heads"], cfg["d_ff"], dropout=cfg["dropout"])
            for _ in range(cfg["enc_s_layers"]))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, keep=None):
        x = F.pad(x, (0, self.n_mels_pad - x.size(-1)))  # input is normalized: 0 == mean
        x = self.patch(x[:, None]).flatten(2).transpose(1, 2)
        x = x + self.pos.to(x.dtype)
        x = gather_tokens(x, keep)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class ActionEncoder(nn.Module):
    """Drumroll (B, T, K_V1) and CC4 (B, T) -> (B, T/patch, d) post-LN tokens."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        p = cfg["action_patch_frames"]
        self.p, self.n_tok = p, SEG_FRAMES // p
        self.roll_proj = nn.Linear(p * K_V1, d)
        self.cc_proj = nn.Conv1d(1, d, kernel_size=p, stride=p)
        self.register_buffer("pos", sincos_1d(d, torch.arange(self.n_tok))[None], persistent=False)
        self.blocks = nn.ModuleList(
            Block(d, cfg["n_heads"], cfg["d_ff"], dropout=cfg["dropout"])
            for _ in range(cfg["enc_a_layers"]))
        self.norm = nn.LayerNorm(d)

    def forward(self, roll, cc, keep=None):
        x = self.roll_proj(roll.reshape(roll.size(0), self.n_tok, -1))
        x = x + self.cc_proj(cc[:, None]).transpose(1, 2)
        x = x + self.pos.to(x.dtype)
        x = gather_tokens(x, keep)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class Predictor(nn.Module):
    """Predicts masked t+1 tokens from [t tokens ; visible t+1 tokens + mask tokens]."""

    def __init__(self, cfg, n_tok, pos, cross):
        super().__init__()
        d = cfg["d_model"]
        self.n_tok = n_tok
        self.register_buffer("pos", pos[None], persistent=False)
        self.seg = nn.Parameter(torch.zeros(2, 1, d))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.seg, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList(
            Block(d, cfg["n_heads"], cfg["d_ff"], cross=cross, dropout=cfg["dropout"])
            for _ in range(cfg["pred_layers"]))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, d)

    def forward(self, z_t, z_t1_vis, vis_idx, ctx=None):
        """z_t: (B, N, d) student t tokens; z_t1_vis: (B, n_vis, d) at `vis_idx`.

        Masked positions get the learned mask token, so the t+1 half is (B, N, d).
        """
        pos, seg = self.pos.to(z_t.dtype), self.seg.to(z_t.dtype)
        cur = self.mask_token.to(z_t.dtype).expand(z_t.size(0), self.n_tok, -1).clone()
        cur = cur.scatter(1, vis_idx[..., None].expand(-1, -1, cur.size(-1)), z_t1_vis)
        x = torch.cat([z_t + pos + seg[0], cur + pos + seg[1]], dim=1)
        for blk in self.blocks:
            x = blk(x, ctx)
        return self.head(self.norm(x[:, self.n_tok:]))


class DrumJEPA(nn.Module):
    """Action-conditioned JEPA over (state, action) segment pairs.

    Args:
      cfg: overrides for DEFAULTS (plain dict, all keys optional).
    """

    def __init__(self, cfg: dict):
        super().__init__()
        cfg = {**DEFAULTS, **(cfg or {})}
        self.cfg = cfg
        self.tau, self.lambda_a, self.mask_ratio = cfg["tau"], cfg["lambda_a"], cfg["mask_ratio"]
        self.mask_blocks, self.mask_scale, self.mask_aspect = (
            cfg["mask_blocks"], cfg["mask_scale"], cfg["mask_aspect"])

        self.Es_stu = StateEncoder(cfg)
        self.Ea_stu = ActionEncoder(cfg)
        self.grid = self.Es_stu.grid
        self.n_s = self.grid[0] * self.grid[1]
        self.n_a = self.Ea_stu.n_tok
        self.n_mask_s = round(self.mask_ratio * self.n_s)
        self.n_mask_a = round(self.mask_ratio * self.n_a)

        self.Es_tea = copy.deepcopy(self.Es_stu).requires_grad_(False)
        self.Ea_tea = copy.deepcopy(self.Ea_stu).requires_grad_(False)

        d = cfg["d_model"]
        self.f = Predictor(cfg, self.n_s, sincos_2d(d, *self.grid), cross=True)
        self.g = Predictor(cfg, self.n_a, sincos_1d(d, torch.arange(self.n_a)), cross=False)
        self._last_masks = None

    # ---- encoders -------------------------------------------------------
    def encode_state(self, x, teacher=False, keep=None):
        """Log-mel (B, SEG_FRAMES, N_MELS) -> (B, n_s, d), or (B, n_keep, d) with `keep`.

        `keep` is a (B, n_keep) index tensor selected after patch- and pos-embedding, so
        the blocks only ever attend over the kept tokens (I-JEPA context encoder).
        """
        return (self.Es_tea if teacher else self.Es_stu)(x, keep)

    def encode_action(self, roll, cc, teacher=False, keep=None):
        """Drumroll (B, T, K_V1) and CC4 (B, T) -> (B, n_a, d), or (B, n_keep, d)."""
        return (self.Ea_tea if teacher else self.Ea_stu)(roll, cc, keep)

    # ---- masks ----------------------------------------------------------
    def _state_mask(self, B, device):
        """Union of `mask_blocks` I-JEPA blocks, trimmed/topped up to n_mask_s."""
        H, W = self.grid
        m = torch.zeros(B, H, W, dtype=torch.bool)
        for b in range(B):
            for _ in range(self.mask_blocks):
                r = torch.rand(2)
                area = (self.mask_scale[0] + r[0].item() *
                        (self.mask_scale[1] - self.mask_scale[0])) * self.n_s
                ar = self.mask_aspect[0] + r[1].item() * (self.mask_aspect[1] - self.mask_aspect[0])
                h = max(1, min(H, round(math.sqrt(area * ar))))
                w = max(1, min(W, round(math.sqrt(area / ar))))
                top = torch.randint(0, H - h + 1, (1,)).item()
                left = torch.randint(0, W - w + 1, (1,)).item()
                m[b, top:top + h, left:left + w] = True
        return self._exact(m.view(B, self.n_s), self.n_mask_s).to(device)

    @staticmethod
    def _exact(m, n_mask):
        """Force each row of a boolean mask to exactly n_mask True, at random."""
        B, N = m.shape
        order = (torch.rand(B, N) + (~m).float()).argsort(dim=1)  # masked first, random within
        out = torch.zeros(B, N, dtype=torch.bool)
        return out.scatter_(1, order[:, :n_mask], True)

    def _action_mask(self, B, device):
        return self._exact(torch.zeros(B, self.n_a, dtype=torch.bool), self.n_mask_a).to(device)

    # ---- loss -----------------------------------------------------------
    @staticmethod
    def _masked_mse(pred, target, mask):
        idx = mask.nonzero(as_tuple=True)
        return F.mse_loss(pred[idx], target[idx])

    @staticmethod
    def _visible_idx(mask):
        """Ascending indices of the unmasked tokens, (B, N - n_mask)."""
        return (~mask).nonzero(as_tuple=True)[1].view(mask.size(0), -1)

    @staticmethod
    def _tok_std(z):
        """Mean over dims of the per-dim std across all tokens in the batch."""
        return z.detach().float().flatten(0, 1).std(dim=0).mean()

    def forward(self, batch: dict) -> dict:
        """One training step's losses and collapse monitors (0-d tensors)."""
        x_t, x_t1 = batch["x_t"], batch["x_t1"]
        B, device = x_t.size(0), x_t.device
        mask_s = self._state_mask(B, device)
        mask_a = self._action_mask(B, device)
        self._last_masks = {"state": mask_s, "action": mask_a}
        vis_s, vis_a = self._visible_idx(mask_s), self._visible_idx(mask_a)

        # Context encodings of t+1 see the visible tokens only; the t side and the
        # action given to f as K/V are full (the action is given, not predicted).
        s_t = self.encode_state(x_t)
        s_t1_vis = self.encode_state(x_t1, keep=vis_s)
        a_t = self.encode_action(batch["a_t"], batch["cc_t"])
        a_t1 = self.encode_action(batch["a_t1"], batch["cc_t1"])
        a_t1_vis = self.encode_action(batch["a_t1"], batch["cc_t1"], keep=vis_a)
        with torch.no_grad():
            s_tgt = self.encode_state(x_t1, teacher=True)
            a_tgt = self.encode_action(batch["a_t1"], batch["cc_t1"], teacher=True)

        loss_s = self._masked_mse(self.f(s_t, s_t1_vis, vis_s, ctx=a_t1), s_tgt, mask_s)
        loss_a = self._masked_mse(self.g(a_t, a_t1_vis, vis_a), a_tgt, mask_a)
        return {
            "loss": loss_s + self.lambda_a * loss_a,
            "loss_s": loss_s.detach(), "loss_a": loss_a.detach(),
            # Student monitors use the full t-side encodings, so they cover all tokens.
            "s_tea_std": self._tok_std(s_tgt), "s_stu_std": self._tok_std(s_t),
            "a_tea_std": self._tok_std(a_tgt), "a_stu_std": self._tok_std(a_t),
        }

    # ---- EMA ------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self):
        """tea <- tau * tea + (1 - tau) * stu for both encoders; once per optimizer step."""
        for stu, tea in ((self.Es_stu, self.Es_tea), (self.Ea_stu, self.Ea_tea)):
            for p_s, p_t in zip(stu.parameters(), tea.parameters()):
                p_t.mul_(self.tau).add_(p_s.detach(), alpha=1 - self.tau)
            for b_s, b_t in zip(stu.buffers(), tea.buffers()):
                b_t.copy_(b_s)

    def n_params(self) -> dict:
        n = {k: sum(p.numel() for p in m.parameters())
             for k, m in (("Es", self.Es_stu), ("Ea", self.Ea_stu), ("f", self.f), ("g", self.g))}
        n["total_student"] = sum(n.values())
        return n


assert N_MELS <= DEFAULTS["n_mels_pad"], "n_mels_pad must cover N_MELS"
