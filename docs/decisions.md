# Decisions

Record of the notable decisions made while merging the three original
repositories into this one (see `history.md` for why they existed
separately in the first place).

## Merge order and method
- Merged in order: gen1 (`1.ML-Cleaner-latest`) → gen2 (`2.Manhwa-Production`)
  → gen3 (`3.Manhwa-cleaner`).
- Built as one linear rebase, not merge commits: each repository's own
  commits were replayed on top of the previous one's tip, so file paths
  accumulate the way they actually did (old prototype files persist as
  history rather than being deleted at the seams).

## Commit versioning scheme
- All commits renumbered to `N.XX.YY`: `N` = generation (1/2/3), `XX` =
  feature number within that generation, `YY` = version of that feature.
  Later simplified from zero-padded (`1.09.01`) to unpadded (`1.9.1`,
  `1.10.1`).
- For gen1, the mapping came directly from the project's existing
  semver-like tags (`v0.1.0`, etc). For gen2/gen3, which had no descriptive
  commit messages at all, feature/version grouping was inferred from diff
  content and commit adjacency.

## File collapsing
- Gen1's separately-numbered file versions were collapsed into single
  evolving files (`remove_manhwa_bg.py`, `cleaner.py`), so history shows
  real diffs between versions instead of separate files sitting side by
  side.

## Data and model removal
- All training-related data (Threshold/ sample PNGs, parameter-search
  reports, evaluation CSVs) and trained model checkpoints (`models/*.pt`,
  `*.json`, versions 1.0-2.1) were removed from every commit in history.
  Reason: those models were trained on copyrighted manhwa chapters; the
  project is moving to an open, reproducible dataset instead. The code that
  produced them was kept.
- `.gitignore` updated afterward so regenerated `models/` and `reports/`
  stay untracked going forward.

## Identity and attribution
- All commit author/committer identity unified to a single name/email,
  replacing the mixed identity used across the original repositories.
- No AI co-authorship or attribution appears anywhere in the history.

## What was kept as-is
- Superseded prototype files (e.g. root-level `remove_manhwa_bg.py`,
  `src/*.py` from gen2) were left in the working tree as historical
  artifacts rather than deleted or archived into a subfolder, to preserve
  the evolution as real, browsable history.
