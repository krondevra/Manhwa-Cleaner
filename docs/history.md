# Project History
This repository is the result of merging three separate git repositories that
were, in practice, one continuous project interrupted multiple times:

1. `1.ML-Cleaner` — rule-based prototypes through the first PyTorch ML pivot
   (generation 1, `1.X.Y` commits)
2. `2.Manhwa-Production` — a parallel continuation with a different file
   layout (`src/`), covering merge/clean/cutframes/montage tooling
   (generation 2, `2.X.Y` commits)
3. `3.Manhwa-cleaner` — the most recent iteration, restructured into
   `tools/`/`docs/`/`scripts/`, with dataset evaluation, active learning and
   parameter-search tooling (generation 3, `3.X.Y` commits)

## Why the project went through three generations
Not a deliberate architectural choice — a consequence of:

- Lack of experience with prototyping/backing up work safely: new attempts
  were started as fresh folders instead of branches or continued work.
- Git was not used actively during early iterations; version control was
  improvised instead via separate numbered files rather than commits.
- Prioritizing getting a result over keeping a clean structure.
- The project was started, stopped, continued, stopped again, and continued
  again over time — each resumption effectively began a new working copy
  rather than picking up the previous one.

## Why older models and data were removed
Generations 1-3 all trained on manhwa chapters the author did not hold rights
to; the resulting model checkpoints (`models/1.0`-`2.1`) and the training
samples/reports behind them (`Threshold/`, `reports/`) existed only for
private prototyping. They were removed from every commit in history because
they should never have been distributable in the first place, not because
the methodology built on them was wrong. The code that produced them stays;
see `decisions.md` for exactly what was removed and how.

## How it was unified
The three repositories were merged into one linear history (this repo), in
order gen1 → gen2 → gen3, with:
- commit messages rewritten to a consistent `N.XX.YY` scheme (see
  `decisions.md`)
- per-tool file versions collapsed into single evolving files (e.g.
  `remove_manhwa_bg.py`)
- author identity unified across all commits
- copyrighted training data and model checkpoints removed from history

See `git log --oneline` for the full resulting timeline.
