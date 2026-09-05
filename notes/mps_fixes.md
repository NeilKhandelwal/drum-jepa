# MPS / Apple-silicon notes

Log every incompatibility and workaround here. You will hit the same ones
again when porting the drum model.

| date | symptom | fix | where |
|---|---|---|---|
| 2026-09-04 | `MPS backend out of memory (62 GiB)` on ViT-S/8 with bs 256 x 4 views at 128 px (1024 images x 256 tokens) | `timm` `set_grad_checkpointing(True)`; recipe unchanged, ~30% more compute. Default on for MPS in the warm-up script. | warmup/lejepa_inet10.py |
| 2026-09-04 | Hard process abort (no Python exception): `MPSNDArrayMatrixMultiplication ... Destination NDArray and Accumulator NDArray cannot have different datatype` | A bf16 activation (from autocast) hit an fp32 `nn.Linear` outside the autocast block. Keep every matmul under `autocast` or cast explicitly; CUDA raises a dtype error here, MPS kills the process. | warmup/lejepa_inet10.py eval loop |
| 2026-09-04 | Throughput: ~4.1 s/step for 1024 images (ViT-S/8, 128 px, bf16, grad ckpt, 8 loader workers) | Not tuned. 100 epochs ~4 h; 800 epochs ~33 h, so the 800-epoch reference does not fit overnight. | warmup/lejepa_inet10.py |
| 2026-09-05 | `--resume` on MPS: `TypeError: RNG state must be a torch.ByteTensor` | `torch.load(..., map_location=dev)` moves the saved RNG ByteTensors to MPS and `torch.set_rng_state` then rejects them. Load the checkpoint with `map_location="cpu"`; `load_state_dict` still puts the parameters on the right device. | scripts/train.py |
| 2026-09-05 | Throughput: 0.45 s/step for the 18.8M drum-JEPA at bs 128 (bf16 autocast, 4 loader workers, 20-step measurement, no grad checkpointing) | Not tuned, and no MPS OOM at this size. 620 steps/epoch on the 6-kit train split -> ~5 min/epoch, ~1.6 h for 20 epochs, an order of magnitude under the CLAUDE.md "1-2 days" estimate. Re-measure over a full epoch before trusting it. | scripts/train.py |
