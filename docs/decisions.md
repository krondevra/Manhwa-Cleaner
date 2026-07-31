# Decisions
Record of the notable decisions made while merging the three original
repositories into this one (see `history.md` for why they existed
separately in the first place).

## Merge order and method
- Merged in order: gen1 (`1.ML-Cleaner`) → gen2 (`2.Manhwa-Production`)
  → gen3 (`3.Manhwa-cleaner`).
- Built as one linear rebase, not merge commits: each repository's own
  commits were replayed on top of the previous one's tip, so file paths
  accumulate the way they actually did (old prototype files persist as
  history rather than being deleted at the seams).

## Commit versioning scheme
- All commits renumbered to `N.XX.YY`: `N` = generation, `XX` =
  feature number within that generation, `YY` = version of that feature.
  No zero-padding (`1.9.1`, `1.10.1`, not `1.09.01`).
- For gen1, the mapping came directly from the project's existing
  semver-like tags (`v0.1.0`, etc). For gen2/gen3, which had no descriptive
  commit messages at all, feature/version grouping was inferred from diff
  content and commit adjacency.
- Bugfixes of an earlier feature reuse that feature's `XX` with `YY+1` —
  they can appear out of `XX` order in the log if a later feature's commits
  land in between (e.g. `4.8.2`/`4.8.3` are fixes of feature `4.8`, logged
  after `4.9.1`).

### Generation boundaries
- **1-2**: the two legacy repos this one was merged from (gen1
  `1.ML-Cleaner`, gen2 `2.Manhwa-Production`), pre-merge.
- **3**: rule-based → classical ML → deep-learning pivot, through model
  12.0's capacity experiment. Ends at `3.48.1`.
- **4**: further training-side levers on the "clauds" bubble-edge defect
  after capacity was ruled out — boundary-loss weighting (13.0), an
  auxiliary SDT head (15.0), scale-match (16.0/17.0), `--repair-frames`
  inference postprocessing, then the CascadePSP/ToonOut refinement era —
  real-manhwa policy clarification, zero-shot probe, Pepper & Carrot
  finetune, RefineHead self-contained architecture, ToonOut/BiRefNet probe.
  Runs `4.1.1`-`4.15.2`, then `4.16.1`-`4.22.12`; closes at commit `4.22.12`
  (2026-07-26) once both third-party-weight options were deliberately
  rejected (licensing provenance, not quality — see
  `ml_strategy_history.md` "Generation 6 pivot").
- **6**: full pivot to a self-synthesized training pipeline — no P&C
  composition tuning, no third-party pretrained weights of any kind. Starts
  at `6.1.1`. Plan: `notes/synthetic_curriculum_plan.md`.

**Generation 5 is reserved and intentionally absent from `main`**: it is
exclusively the `archive` branch's label for these same CascadePSP/ToonOut
commits under their original numbers `5.1.1`-`5.7.12` (tip `130ad9f`,
content-identical to `4.16.1`-`4.22.12` above) — preserved unchanged there
while `main`'s copies were folded into generation 4 in the 2026-08-10
consolidation (message + this section's own text renumbered; no code or
checkpoint content touched).

### 2026-07-23 history rewrite
23 commits made between `3.48.1` and the CascadePSP work had never been
given version prefixes (an oversight, not a deliberate change of
convention). Rewritten in place via `git filter-branch --msg-filter` to
apply the scheme retroactively, splitting the ungrouped run into
generation 4 per the boundaries above. Old pre-rewrite hashes are
preserved on branch `backup/pre-restructure-2026-07-23` — nothing was
deleted, only the commit messages changed (content diff between the
backup branch and the rewritten history is empty). Full record:
`notes/restructure_2026-07-23_plan.md`. Reference commits by version
number from this point forward, not by hash, since a rewrite changes every
descendant hash.

### 2026-08-01: `N.X` two-part break (`6.4`-`6.7`), not rewritten
Every commit from `1.1.1` through `6.3.3` (`ef59b59`) follows the
three-part `N.XX.YY` scheme with no exceptions. Starting at `5c8d28f`
("`6.4`: halo defect investigation..."), the `.YY` component was dropped
entirely — not a skipped number, a structural regression to bare `N.X` —
and the break persisted uncorrected through `6.5`, `6.6`, and `6.7`
(current HEAD as of this note). Per policy, **this is not being rewritten**
(unlike the 2026-07-23 gap above, which was a mechanical oversight caught
immediately; this one shipped and was built on top of). Going forward:
- Three-part `N.XX.YY` resumes starting with the next commit.
- `6.4`-`6.7` are treated as implicitly single-version entries (`XX` = 4,
  5, 6, 7 respectively, `YY` = 1 each) for the sole purpose of deriving
  future numbers — i.e. the next brand-new feature commit is `6.8.1`; a
  direct follow-up specifically to the `6.7` cleanup work would be `6.7.2`.
- This mirrors the existing "bugfixes reuse `XX` with `YY+1`" rule above;
  no new rule was invented, the two-part commits are just read as if their
  omitted `.1` were implicit.

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

## Model file naming convention
Separate from the `N.XX.YY` **commit-message** scheme above — this is a
different numbering system for `data/models/` **files**, don't conflate
the two.
- Sequential `N.0` in `data/models/`, continuing from the last used number.
  `1.0` through `17.0` cover generations 1-5; `18.0` starts the
  generation-6 synthetic-pivot era (`18.0-frames.pt`, promoted from
  `.tmp/checkpoints/stage1/a6_full10k/a6_full10k.pt`, the checkpoint
  documented as the standing Stage 1 baseline in `ml_strategy_history.md`
  and `notes/synthetic_curriculum_plan.md` — not
  `data/models/gen6-stage1.pt`, an earlier, pre-bugfix run of the same
  stage that was never adopted and has been deleted).
- A descriptive suffix (`-frames`, `-bubbles`, `-sfx`, ...) is added only
  when a checkpoint needs distinguishing, naming the single thing that
  version isolates — same pattern as existing suffixes: `-strips`
  (manhwa-scroll dataset restructuring), `-boundaryloss` (boundary-weighted
  loss), `-sdt` (auxiliary signed-distance-transform head), `-baseline`
  (recipe-simplification control run). Plain `N.0` with no suffix is for
  checkpoints that don't introduce a new named experimental variable
  relative to the prior one. Established pattern for the gen-6 stage
  checkpoints specifically: `18.0-frames` (Stage 1), `18.1-bubbles` (Stage
  2, once promoted), `18.2-sfx` (Stage 3, once promoted) — not created yet,
  Stage 2/3 aren't adopted baselines as of this note.
- Per-epoch training savepoints (`*.epochN.pt`) never live in
  `data/models/` — they belong under `.tmp/checkpoints/<name>/`, alongside
  the full training history for that run (see the 2026-08 `.tmp/`
  restructuring). `data/models/` holds only the one promoted, versioned
  checkpoint per release.
- `cascadepsp-*` and `black-1.0` are deliberately separate naming
  lineages, not part of the main `N.0` sequence: `cascadepsp-*` is
  excluded from git tracking entirely (`.gitignore`: exceeds GitHub's
  100MB limit, and was rejected on licensing-provenance grounds regardless
  of measured quality); `black-1.0` was called out in
  `ml_strategy_history.md` as a "new naming lineage, not part of the
  white-bg `N.0` series" when it was created, since it's trained on a
  disjoint black-background-only composition rather than being a variant
  of the white-bg recipe the `N.0` sequence otherwise tracks.

## Identity and attribution
- All commit author/committer identity unified to a single name/email,
  replacing the mixed identity used across the original repositories.
- No AI co-authorship or attribution appears anywhere in the history.

## What was kept as-is
- Superseded prototype files (e.g. root-level `remove_manhwa_bg.py`,
  `src/*.py` from gen2) were left in the working tree as historical
  artifacts rather than deleted or archived into a subfolder, to preserve
  the evolution as real, browsable history.

## Which architectural solutions proved successful, and which were discarded
Discarded:
- Pure flood-fill from image edges (v1-v2): destroys white content trapped
  inside frames/speech bubbles once it connects to the edge; the same pixel
  color can be background or content depending on structure alone, which
  flood-fill cannot see.
- Panel detection via black horizontal lines, row-based restore, magic-wand
  imitation (v2-v3): closer to the manual result but still consistently
  worse than a manually cleaned reference; abandoned as a dead end rather
  than kept as "good enough."
- OpenCV Random Trees pixel classifier (v4-v7): trained on a single
  original/cleaned image pair. Looked good only because it exact-copied the
  training image's own alpha channel back out — not evidence of learning.
  Failed on unseen chapters (poor quality, ~11 minutes per chapter) and was
  dropped in favor of real segmentation.

Successful:
- Supervised binary segmentation with a small U-Net (PyTorch), 7-channel
  input: RGB plus threshold/morphological-open/morphological-close/Canny
  guidance channels that directly encode the manual Photoshop workflow this
  project automates (threshold ~90, min/max radius 2px, magic wand). This
  was the actual turning point from "looks plausible" to "generalizes."
- Dataset quality over quantity: the jump from model line 1.x to 2.0 came
  from a few carefully and consistently cleaned chapters, not from adding
  more inconsistent ones.
- Heuristic evaluation without ground truth, used to mine hard cases instead
  of guessing which chapters to add next; an active learning loop (clean →
  train → test on unseen → fix failures → repeat); and semi-automatic
  mask/ROI generation plus Photoshop-style parameter search (separate
  black/white, hard/soft profiles) for hard cases such as black backgrounds.
