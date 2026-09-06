"""Amortized inverse model for E4: recover the action from a state transition.

h reads frozen state embeddings of x_t and x_{t+1} (one token grid each, plus the
encoder's fixed 2-D sin-cos positions and a learned segment embedding) and, in the
`use_at` variant, the previous action a_t as one token per 250 ms step. A small
pre-norm transformer mixes them; `n_steps` learned query tokens then cross-attend
to that sequence and each decodes its own 25-frame chunk of a_{t+1}: onset logits,
velocities, and the hi-hat pedal track cc_{t+1}.

The blocks are drumjepa.model.Block, so the cross-attention layers also self-attend
over the 8 queries. That is a small deviation from "two cross-attention layers" and
costs ~0.5M parameters, but it reuses the tested block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from drumjepa.drum_map import K_V1
from drumjepa.features import SEG_FRAMES
from drumjepa.model import Block, sincos_1d, sincos_2d


class InverseModel(nn.Module):
    """h(s_t, s_{t+1}[, a_t]) -> the action of the t+1 segment.

    Args:
      d_in: width of the input tokens (256 for a JEPA encoder, 375 for raw mel patches).
      use_at: also read the previous action (a_t, cc_t).
      grid: (time, freq) token grid of the state encoder; n_tok is its product and the
        time axis sets the number of decoded steps.
      d_model, d_ff, n_heads, n_layers, n_cross, dropout: h's own size.
    """

    def __init__(self, d_in=256, use_at=False, grid=(8, 16), d_model=256, d_ff=512,
                 n_heads=4, n_layers=3, n_cross=2, dropout=0.0):
        super().__init__()
        d = d_model
        self.use_at, self.grid = use_at, tuple(grid)
        self.n_tok = grid[0] * grid[1]
        self.n_steps = grid[0]
        self.step_frames = SEG_FRAMES // self.n_steps
        assert self.n_steps * self.step_frames == SEG_FRAMES, "grid time axis must divide SEG_FRAMES"

        self.in_proj = nn.Linear(d_in, d)
        self.seg = nn.Parameter(torch.zeros(3, 1, d))  # s_t, s_{t+1}, a_t
        self.query = nn.Parameter(torch.zeros(1, self.n_steps, d))
        nn.init.trunc_normal_(self.seg, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)
        self.register_buffer("pos_s", sincos_2d(d, *self.grid)[None], persistent=False)
        self.register_buffer("pos_a", sincos_1d(d, torch.arange(self.n_steps))[None],
                             persistent=False)
        if use_at:
            self.act_proj = nn.Linear(self.step_frames * (K_V1 + 1), d)
        self.blocks = nn.ModuleList(Block(d, n_heads, d_ff, dropout=dropout)
                                    for _ in range(n_layers))
        self.cross = nn.ModuleList(Block(d, n_heads, d_ff, cross=True, dropout=dropout)
                                   for _ in range(n_cross))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, self.step_frames * (2 * K_V1 + 1))

    def forward(self, z_t, z_t1, a_t=None, cc_t=None) -> dict:
        """Frozen tokens (B, n_tok, d_in) x2 -> onset logits, velocities, CC4.

        Args:
          z_t, z_t1: frozen state embeddings of x_t and x_{t+1}.
          a_t, cc_t: previous drumroll (B, SEG_FRAMES, K_V1) and pedal track
            (B, SEG_FRAMES); required when `use_at`, ignored otherwise.

        Returns:
          dict with `onset_logit` and `velocity` (B, SEG_FRAMES, K_V1) and `cc`
          (B, SEG_FRAMES).
        """
        B = z_t.size(0)
        pos, seg = self.pos_s.to(z_t.dtype), self.seg.to(z_t.dtype)
        x = torch.cat([self.in_proj(z_t) + pos + seg[0],
                       self.in_proj(z_t1) + pos + seg[1]], dim=1)
        pos_a = self.pos_a.to(x.dtype)
        if self.use_at:
            assert a_t is not None and cc_t is not None, "use_at=True needs a_t and cc_t"
            chunks = torch.cat([a_t.reshape(B, self.n_steps, self.step_frames * K_V1),
                                cc_t.reshape(B, self.n_steps, self.step_frames)], dim=-1)
            x = torch.cat([x, self.act_proj(chunks.to(x.dtype)) + pos_a + seg[2]], dim=1)
        for blk in self.blocks:
            x = blk(x)
        q = self.query.to(x.dtype).expand(B, -1, -1) + pos_a
        for blk in self.cross:
            q = blk(q, x)
        out = self.head(self.norm(q)).view(B, SEG_FRAMES, 2 * K_V1 + 1)
        return {"onset_logit": out[..., :K_V1], "velocity": out[..., K_V1:2 * K_V1],
                "cc": out[..., -1]}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def inverse_loss(out: dict, roll, cc, pos_weight) -> dict:
    """Onset BCE (pos-weighted) + velocity MSE at true onsets + CC4 MSE.

    Velocity is only supervised where an onset actually happened: elsewhere the
    target cell is 0 and would drag every velocity toward silence.

    Args:
      out: `InverseModel.forward` output.
      roll: true drumroll (B, SEG_FRAMES, K_V1), velocity/127 in onset cells.
      cc: true pedal track (B, SEG_FRAMES), CC4/127.
      pos_weight: scalar tensor, the BCE weight on positive cells.
    """
    logit = out["onset_logit"]
    onset = (roll > 0).to(logit.dtype)
    bce = F.binary_cross_entropy_with_logits(logit, onset, pos_weight=pos_weight.to(logit.dtype))
    vel = ((out["velocity"].float() - roll.float()) ** 2 * onset.float()).sum() / onset.sum().clamp(min=1)
    cc_mse = F.mse_loss(out["cc"].float(), cc.float())
    return {"loss": bce + vel + cc_mse, "bce": bce.detach(),
            "vel": vel.detach(), "cc": cc_mse.detach()}
