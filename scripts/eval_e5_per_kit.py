"""Per-kit E5 linear onset F1 from a run's cached E5 features (runs/<run>/e5/feat_*.npz).

Same protocol as eval_e5.py's linear onset probe: StandardScaler + 14 independent logistic
regressions fit on the train-split clips of the train kits, scored on the test split at
threshold 0.5, but reported per kit instead of per kit group, for the drumjepa and
raw_mel representations. The unweighted mean over held-out kits reproduces eval_e5's
held-out macro-F1 to within 0.01 (docs/k12.md).

    .venv/bin/python scripts/eval_e5_per_kit.py drumjepa_v2_12k drumjepa_v2_12k_auxrec01
"""
import json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def f1(y, p):
    tp = (y & p).sum(0); fp = (~y & p).sum(0); fn = (y & ~p).sum(0)
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)

for run in sys.argv[1:]:
    r = json.load(open(f"runs/{run}/e5/results.json"))
    kits, train = r["kits"], set(r["train_kits"])
    tr = np.load(f"runs/{run}/e5/feat_train.npz"); te = np.load(f"runs/{run}/e5/feat_test.npz")
    train_ids = [i for i, k in enumerate(kits) if k in train]
    mtr = np.isin(tr["kit_id"], train_ids)
    Ytr = tr["onset"][mtr].reshape(-1, 14)
    out = {}
    for rep in ("drumjepa", "raw_mel"):
        d = tr[rep].shape[-1]
        sc = StandardScaler().fit(tr[rep][mtr].reshape(-1, d).astype(np.float32))
        Xtr = sc.transform(tr[rep][mtr].reshape(-1, d).astype(np.float32))
        clfs = [LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000).fit(Xtr, Ytr[:, k])
                if 0 < Ytr[:, k].mean() < 1 else float(Ytr[:, k].mean()) for k in range(14)]
        per = {}
        for i, k in enumerate(kits):
            m = te["kit_id"] == i
            X = sc.transform(te[rep][m].reshape(-1, d).astype(np.float32))
            Y = te["onset"][m].reshape(-1, 14)
            P = np.column_stack([np.full(len(X), c) if isinstance(c, float) else c.predict_proba(X)[:, 1] for c in clfs]) >= 0.5
            per[k] = float(f1(Y, P).mean())
        out[rep] = per
    print(f"## {run}")
    print("| kit | group | drumjepa | raw_mel |"); print("|---|---|---|---|")
    for k in kits:
        print(f"| {k} | {'train' if k in train else 'HELD-OUT'} | {out['drumjepa'][k]:.3f} | {out['raw_mel'][k]:.3f} |")
    ho = [k for k in kits if k not in train]
    print(f"held-out mean: drumjepa {np.mean([out['drumjepa'][k] for k in ho]):.3f}, raw_mel {np.mean([out['raw_mel'][k] for k in ho]):.3f}", flush=True)
