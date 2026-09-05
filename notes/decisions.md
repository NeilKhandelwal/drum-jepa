# Decisions on details the paper leaves unspecified

One entry per decision. Say what was chosen, why, and what could go wrong.

## Mel normalization (2026-09-05)
`SegmentPairs` reads `mel_mean`/`mel_std` from `<cache_dir>/stats.json` by default
(train split, 1M random frames). Chosen so no caller can forget to normalize.

Watch for: the stats belong to the cache build, not the run. Rebuilding the cache,
changing the train kit subset, or pointing at a different cache directory changes
the normalization silently, and checkpoints trained under one set of stats are not
comparable under another. Every run config must record the `mel_mean`/`mel_std`
actually used, and eval code must load the checkpoint's values rather than the
current `stats.json`.
