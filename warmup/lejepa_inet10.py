"""LeJEPA minimal example (ViT-S/8, Imagenette), device-agnostic.

Port of third_party/lejepa/MINIMAL.md with the same hyperparameters and loss,
minus hydra, wandb and HuggingFace datasets. Runs on CUDA, MPS or CPU.

Purpose: a clean monotone training curve and MPS experience before porting
the drum model. Reference: 90.7% top-1 online linear probe at 800 epochs
(one GPU, ~200 min).

    python warmup/lejepa_inet10.py --epochs 1 --max-steps 20     # smoke test
    python warmup/lejepa_inet10.py --epochs 100
    python warmup/lejepa_inet10.py --epochs 800

Logs one row per epoch to <run-dir>/metrics.csv and saves <run-dir>/last.pt.

Dataset: Imagenette 160px under --data. If torchvision's downloader fails with
an SSL certificate error (python.org builds), fetch it by hand:
    curl -L -o data/imagenette/imagenette2-160.tgz \
      https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
    tar -xzf data/imagenette/imagenette2-160.tgz -C data/imagenette
"""
import argparse
import csv
import os
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision.datasets import Imagenette
from torchvision.ops import MLP
from torchvision.transforms import v2

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularizer (Epps-Pulley on random 1-D slices)."""

    def __init__(self, knots=17, slices=256):
        super().__init__()
        self.slices = slices
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):  # proj: (V, N, D)
        A = torch.randn(proj.size(-1), self.slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t          # (V, N, slices, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)  # (V, slices)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=16, img_size=128):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_small_patch8_224", pretrained=False, num_classes=512,
            drop_path_rate=0.1, img_size=img_size)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):  # x: (N, V, C, H, W)
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class MultiView(torch.utils.data.Dataset):
    def __init__(self, root, split, V=1, img_size=128):
        self.V = V
        self.ds = Imagenette(root, split=split, size="160px",
                             download=not os.path.isdir(os.path.join(root, "imagenette2-160")))
        norm = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
        self.aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(), *norm])
        self.test = v2.Compose([v2.Resize(img_size), v2.CenterCrop(img_size), *norm])

    def __getitem__(self, i):
        img, y = self.ds[i]
        img = img.convert("RGB")
        tf = self.aug if self.V > 1 else self.test
        return torch.stack([tf(img) for _ in range(self.V)]), y

    def __len__(self):
        return len(self.ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/imagenette")
    ap.add_argument("--run-dir", default=None, help="default: runs/lejepa_inet10_e<epochs>")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke test)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--V", type=int, default=4)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--lamb", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=None,
                    help="activation checkpointing in the backbone (default: on for MPS)")
    args = ap.parse_args()

    dev = pick_device()
    if args.grad_ckpt is None:
        args.grad_ckpt = dev.type == "mps"  # bs*V=1024 images of 256 tokens OOMs at 62 GB otherwise
    amp = not args.no_amp and dev.type != "cpu"
    run_dir = args.run_dir or f"runs/lejepa_inet10_e{args.epochs}"
    os.makedirs(run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    print(f"device={dev} amp(bf16)={amp} grad_ckpt={args.grad_ckpt} run_dir={run_dir}")

    train_ds = MultiView(args.data, "train", V=args.V)
    test_ds = MultiView(args.data, "val", V=1)
    pw = args.workers > 0
    train = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=True,
                       num_workers=args.workers, persistent_workers=pw)
    test = DataLoader(test_ds, batch_size=256, num_workers=args.workers, persistent_workers=pw)

    net = ViTEncoder(proj_dim=args.proj_dim).to(dev)
    net.backbone.set_grad_checkpointing(args.grad_ckpt)
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 10)).to(dev)
    sigreg = SIGReg().to(dev)
    n_params = sum(p.numel() for p in net.backbone.parameters())
    print(f"backbone params: {n_params/1e6:.1f}M  train={len(train_ds)} test={len(test_ds)}")

    g1 = {"params": net.parameters(), "lr": args.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])
    warmup_steps = len(train)
    total_steps = len(train) * args.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=1e-3)
    sched = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    fields = ["epoch", "step", "train/inv", "train/sigreg", "train/lejepa", "train/probe",
              "test/acc", "test/emb_std", "lr", "epoch_s"]
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    step = 0
    for epoch in range(args.epochs):
        net.train(), probe.train()
        t0 = time.time()
        sums = {k: 0.0 for k in ("inv", "sigreg", "lejepa", "probe")}
        n = 0
        for vs, y in tqdm.tqdm(train, total=len(train), desc=f"epoch {epoch}", leave=False):
            vs, y = vs.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * args.lamb + inv_loss * (1 - args.lamb)
                y_rep, yhat = y.repeat_interleave(args.V), probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)
                loss = lejepa_loss + probe_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            for k, v in zip(sums, (inv_loss, sigreg_loss, lejepa_loss, probe_loss)):
                sums[k] += v.item()
            n += 1
            if args.max_steps and step >= args.max_steps:
                break

        net.eval(), probe.eval()
        correct, embs = 0, []
        with torch.inference_mode():
            for vs, y in test:
                vs, y = vs.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                with autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    emb = net(vs)[0]
                    yhat = probe(emb)  # keep inside autocast: bf16 x fp32 matmul aborts MPS
                correct += (yhat.argmax(1) == y).sum().item()
                embs.append(emb.float().cpu())
        emb_std = torch.cat(embs).std(0).mean().item()  # collapse check: shrinks -> collapse
        row = {"epoch": epoch, "step": step, "test/acc": correct / len(test_ds),
               "test/emb_std": emb_std, "lr": opt.param_groups[0]["lr"],
               "epoch_s": round(time.time() - t0, 1),
               **{f"train/{k}": v / max(n, 1) for k, v in sums.items()}}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        torch.save({"net": net.state_dict(), "probe": probe.state_dict(), "epoch": epoch,
                    "args": vars(args)}, os.path.join(run_dir, "last.pt"))
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
