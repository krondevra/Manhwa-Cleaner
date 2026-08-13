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
  at `6.1.1`. Plan: `notes/synthetic_curriculum_plan.md`. Concluded at the
  measured ~12.5% semantic floor (Recipe A, `6.9.0`-`6.9.2`); ends at `6.9.2`.
- **7**: classical spiky-cloud replication track — the manual Photoshop
  pipeline decoded and mechanized as deterministic OpenCV code (per-plan
  rounds v12-v27), consolidated into `src/spiky/pipeline.py`, plus the
  2026-08-10 repository cleanup. Runs `7.1.1`-`7.13.5`.
- **8**: modular detector-framework architecture — classical rules/geometry
  only (no ML by deliberate choice, not as a temporary constraint), existing
  validated logic EXTRACTED into `src/classifiers/` modules rather than
  rewritten; delete-background-over-preserve-frame bias at decision
  boundaries. Starts at `8.1.1`, developed on the `testing` branch, merged
  to `main` per-classifier only with explicit user confirmation.

**Generation 5 is reserved and intentionally absent**: the CascadePSP/ToonOut
commits originally numbered `5.1.1`-`5.7.12` were folded into generation 4 as
`4.16.1`-`4.22.12` in the 2026-08-10 consolidation (messages + this section's
own text renumbered; no code or checkpoint content touched). A local `archive`
branch that had bookmarked the era under its original numbers was deleted the
same day once confirmed to be a plain ancestor of `main` (zero unique commits)
— the full history, content-identical, lives on `main`; the number 5 is simply
never reused.

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

## Classical replication track (2026-08-07, plans v12+) — attempt log

The manual pipeline-v2 procedure was replicated as deterministic classical CV
(`.tmp/scripts-manual/replicate_pipeline_v{2,3,4}.py`); per this track's rule, every
mechanism attempt is logged here with its measured result:

- **v13 leak attempt 1 — near-black-stroke barrier: NO-OP** (2026-08-07). Subtracting
  gray<=40 strokes from the soft-white mask changes nothing: those pixels are already below
  the mask's own threshold (~43). Floods cross BRIGHT bridges (steam, anti-aliasing gaps),
  never dark pixels. Kept as the diagnosis that redirected the fix.
- **v13 leak attempt 2 — Canny(60,120) barrier on soft-white: ADOPTED** (2026-08-07).
  −52% detected border-crossing leakage, 5/7 gold parts to zero, over-deletion down on every
  part, under-del cost <=0.06pp/part.
- **v13 leak attempt 3 — leak-detector post-filter: ADOPTED** (2026-08-07). The barrier-split
  detector validated at precision 1.000 (~280k flagged px, zero false positives) before use;
  subtracting its detections removed the residual 135k px at zero under-del cost. Result:
  zero detected border-crossing leakage on all 7 gold parts (v3).
- **v14 seed attempt 1a — barrier-split the hard mask, rule unchanged: REJECTED** (2026-08-07).
  Split-off pure-white panel interiors get seeded WHOLESALE -> over-del guard violated
  (fit page over 0.18%->2.18%; 001-3 +0.51pp).
- **v14 seed attempt 1b — barrier-split + gray>=250-GATED seeding + frac250 relaxed to 0.90:
  ADOPTED as v4** (2026-08-07; the 2x2 factorial was COMPLETED 2026-08-08 per the
  one-variable rule — 1b changed two coupled knobs, so the two missing quadrants were run on
  the same harness/pages). Gold-part white-track error median 8.77%->3.22%, worst
  16.85%->8.35%, over-del DOWN on every gold part, leakage stays zero. **Factorial
  attribution (measured)**: (gated, 0.981) = gating ALONE cuts over-del (001-3 2.08->0.64%,
  fit 2.18->1.99%) with ZERO recall change (001-2 under-white 11.65% vs 1a's 11.63%);
  (wholesale, 0.90) = relaxation ALONE delivers nearly all of 1b's recall (001-2 6.59%,
  002-1 1.03%, 033-1 0.70%) at +1-2pp over-del per part — NOT catastrophic as predicted
  (prediction was overcautious; recorded). So: relaxation carries the recall gains, gating
  carries the over-del safety; 1b composes both, and remains the best quadrant. Flagged
  cost: the fit page's judgment-class document panel is re-exposed (+1.96pp over there; page
  total 2.65%, still within its 3.0% bar) — bright-white-content panels with broken frame
  rings remain the track's top judgment-class defect, matching the original v1-v2 flood-fill
  lesson above (structure, not color, decides).
- **v14 seed attempt 1c — stronger split barrier Canny(30,90): NO-OP** (2026-08-07).
  Byte-identical results to 1b; the 60/120 edges already split everything splittable.
- **v14 new-class attempt — page-adaptive white-band height rule: REJECTED, decisive**
  (2026-08-07). Against the white-panel-interior ambiguity surfaced by the synthetic breadth
  test (a full-bleed panel's pure-white interior is locally identical to a gutter: white band
  between two full-width black bars — the original v1-v2 flood-fill lesson in classical form,
  and NOT a v4 regression: v3 fails the same 5 synthetic pages WORSE, 17-44% vs 15-20%).
  Excluding tall-outlier bands (K x median height, K swept 1.5/2.0/3.0) fixed NONE of the
  synthetic outliers (their interiors don't register as uniform white bands) and
  catastrophically broke real pages (+31-39pp under-deletion — real gutters became "tall
  outliers" against text-gap-dominated medians). The demonstrated cross-distribution
  fragility is itself the evidence that layout-statistics heuristics won't hold this class;
  the information is structural/semantic, not local. Recommended next step (NOT launched,
  needs explicit go-ahead): the scoped micro-net local classifier from the v14 brief,
  synthetic-only training per project rules.
- **v15 Track 2 — soft-gradient barrier: ADOPTED into v5** (2026-08-07). 001-1's
  "white-pockets" miss re-diagnosed: NOT dark-layout pockets and NOT band-ambiguity — the
  residual seed-merge failure at page scale (title-transition fades merge white background
  into 2.5M-px components at frac250 0.36-0.44, below any workable bar; Canny can't cut soft
  gradients). Fix: add |grad(Gaussian(gray, sigma 8))| > 1.0 to the hard-mask split barrier.
  Measured: 001-1 under-white 8.33%->1.14%, 001-2 6.62%->0.50%, fit page total -0.83pp.
  Flagged: 001-1 over-del +0.85pp exceeds the +0.3pp guard letter — accepted on the 8.5:1
  exchange with visual confirmation the new over-del is the KNOWN white-inside-panel
  ambiguity class, not new damage.
- **v15 Track 1 — micro band classifier (BandNet, 24,691 params): HONEST NEGATIVE; hook
  shipped OFF by default** (2026-08-07; attempt log corrected 2026-08-07 per the
  one-variable-per-run discipline — the original "attempt 2" conflated data scale AND loss
  balancing in one run; both variables were re-run in isolation on the SAME data/seeds, and
  the record below is the corrected, isolated version. The isolated numbers CHANGED THE
  ATTRIBUTION but NOT the verdict).
  Attempt 1 (250 pages, plain BCE, lr 1e-3, save-last): frozen at the 9:1 class prior
  (val_acc 0.9149 = the gutter fraction).
  Attempt 2a (loss ISOLATED: 250 pages + pos_weight, all else = attempt 1): breaks the
  prior-freeze immediately (interior-recall 1.000 from epoch 1) but OSCILLATES violently
  (gutter-recall 0.000-0.965 across epochs) — **the oscillation is caused by the loss
  balancing, not the data scale**.
  Attempt 2b (data ISOLATED: 1k pages, plain BCE): prior-frozen for ~6 epochs, then escapes
  SLOWLY without any balancing, reaching balanced 0.849 by epoch 15 (gutter 0.785 / interior
  0.913) — **data scale alone was nearly sufficient given enough epochs; balancing bought
  speed at the cost of stability** (this corrects the original entry's "imbalance + data
  starvation" framing).
  Attempt 3 (lr 3e-4 + StepLR decay + best-checkpoint selection, 1k + pos_weight): STABLE,
  val balanced-acc 0.874 (interior 0.957 / gutter 0.792). The lr/scheduler bundle was
  ISOLATED 2026-08-08 (same data/seeds): **3-lr (lr 3e-4 alone, no scheduler) is stable and
  reaches balanced 0.870 — the lr reduction alone delivers the stability; 3-sched (lr 1e-3 +
  StepLR alone) still crashes early (epoch 7: balanced 0.545, interior 0.109) and only
  settles once the decay has effectively reached the low lr — the scheduler is redundant
  polish.** Best-checkpoint selection needs no isolation run: it is measurement bookkeeping
  (which artifact is kept), and every log records per-epoch metrics from which best and last
  are both readable.
  VERDICT (unchanged, now verified on every variant incl. the lr/scheduler isolations): the
  integration's cost structure is asymmetric — protecting one misclassified gutter band
  costs its whole area — requiring gutter-recall >= 0.995, and the operating-point sweep
  shows interior-recall collapsing there on ALL trained variants: attempt 3 best 0.00-0.11;
  isolated 2a 0.000-0.125 at gutter-recall 1.0; isolated 2b 0.109 at 1.0 (0.283 at 0.991);
  3-lr and 3-sched checkpoints swept with the same collapse shape. NO viable operating
  point, regardless of which variable is isolated. Battery with hook ON: fixed 2/5 target pages,
  collapsed 7 previously-passing synthetic pages to 15-33% under-deletion. Class stays OPEN;
  checkpoints kept in .tmp for future experiments (richer context is the untested axis).
- **v16 Cluster 1 — two-band calibration: ADOPTED (both bands measured, separate as
  predicted)** (2026-08-07). (a) JPEG border residue measured at [240,250) (76.6% of
  border-adjacent undeleted px >= 240) -> border sweep, 3px radius, 3 iterations. (b)
  under-frame gray line measured at 210-220 (2,693/3,910 px in the 333-region histogram) ->
  [200,230] band deleted only directly below near-black strokes AND near deleted background.
  No shared-knob tension in practice: different bands, different triggers.
- **v16 Cluster 2 — variable-position edge line: MECHANISM REUSED from Cluster 3** (2026-08-07).
  Found at col 688 (1-3px from right edge, 58-62k rows, both chapters) + smaller left-edge
  runs on 007 (position varies, as the user said). One thin-line exemption in the local
  reclaim (width <= 4px bypasses the pocket-area cap when edge-touching): residual 3,181 ->
  68 px on the test strip. No parallel implementation built.
- **v16 Cluster 3 — local background reclaim: attempts 1+2 ADOPTED, attempt 3 REJECTED**
  (2026-08-07). Core: bright pockets (>=240) with ring >= 85% ink+deleted(+page-edge), pocket
  area < 10,000 (repair_frame_interiors' own interior convention = the 3(d) size guard),
  closed-frame interiors excluded. Attempt-2 fix after the first adversarial test leaked
  1,941 px of text counters: OUTER-ring guard (context beyond the enclosing ink must be
  >= 35% deleted — an SFX glyph floats on deleted field; letter counters in kept text do
  not). Corrected adversarial test (counters inside correctly-KEPT bubbles, two regions):
  0 suspicious px both ways, SFX pockets reclaim 4.5k px. (The original doc-panel negative
  was a test-design artifact: v5 already wrongly deletes that panel, so step C merely joined
  a pre-existing judgment-class deletion.) Attempt 3 (2px ink dilation in ring composition,
  targeting the 666-class note-text pockets): REJECTED — bought -3% residual there while
  leaking 9,365 px into kept bubble-text counters. 666-class large fragmented pockets remain
  a documented residual.
- **v16 per-step battery attribution — COMPLETED 2026-08-08** (the A/B/C steps had entered
  the regression battery jointly; the ladder v5 -> +C -> +C+A -> +C+A+B was run through the
  full battery). Marginal deltas on gold white-track: **C** = small improvements or neutral
  everywhere (-0.061 to +0.034pp/part; fit page -0.021pp); **A** = mixed-small (-0.172 to
  +0.069pp/part; fit page -0.133pp — A removes JPEG residue the fit page counted as
  under-white); **B** = near-neutral on battery pages (<=0.01pp everywhere; its evidence is
  its measured target examples — battery pages carry few under-frame lines). EVERY rung
  individually passes all guards (no part regresses > 0.3pp at any step; synthetic
  passing-15 stays clean at every rung) — the v16 adoption stands with per-step attribution.
- **v19 issue 2 classification — border artifacts: RELATED-BUT-DISTINCT band, new mechanism
  E** (2026-08-08). The 007 etalon triplet (`.tmp/minmax/`, alpha-decoded: ZERO over-deletion
  vs the etalon on the crop, 4,884 px under-deletion in three classes) shows the "grayish
  line" strip at gray p50=191 — BELOW v16 step B's [200,230] floor — and on the BACKGROUND
  side of the frame line, which B's below-only trigger cannot reach. Not a regression of the
  v16 fix; a distinct band+geometry.
- **v19 E (frame strip): E3 ADOPTED into v7** (2026-08-08; 3 attempts). E1 (any-dark-within-
  4px) -486/+455 on the etalon crop — rejected, cost equals benefit; E2 (long-horizontal-run
  trigger >=100px) -334/+72; E3 (E2 + geodesic reachability from deleted background through
  the [185,230] band, <=8px) -334 under / ZERO over on the crop, page-wide flagged pixels
  visually verified as frame-adjacent background AA. Battery: every gold part improves
  slightly (deltas -0.0002..-0.0089pp), fit +0.0005pp, synth clean.
- **v19 D (floating-text halo sweep, the user's manual "select by color threshold 23"
  recipe mechanized): HONEST NEGATIVE, 3 attempts** (2026-08-08). D1 blur-context (17.2k
  suspicious px on 007 — symmetric context bites panel interiors near borders); D2
  geodesic-25 (301k — 25px bright-erosion ribbons along every inkless art/background
  boundary); D3 geodesic-6 + 2k-px component cap (87k — ribbons arrive in cap-sized
  chunks). Root cause of failure: the manual recipe works because a HUMAN selects the
  floating-text region first; that locality judgment has no safe classical proxy found.
  Etalon upside forgone: ~2.2k px on the reference crop. Class stays open.
- **v19 F (page-edge fused-gap pockets, the v16 222-class): HONEST NEGATIVE, 3 attempts**
  (2026-08-08). F1 ring-0.60 (-423 px, but 35.5k suspicious page-wide — the edge-contact
  test is trivially satisfied by full-height margin deletions); F2 ring-0.55 (-663, same
  flaw); F3 margin-adjacency (safe but captures -5 px — the fused gap does not actually
  touch the deleted margin). No variant is both safe and effective. Class stays open; the
  SFX-punch-through guard was never violated by the shipped config (E-only adds no
  deletions beyond frame-adjacent AA strips).
- **v20 region-selection round (D floating-text halos, F edge fused gaps): candidate
  generation SOLVED classically, accept decision FAILED both classically and with scoped
  synthetic-trained classifiers — NO SHIP, both classes to the manual/GUI track**
  (2026-08-08). Generation (`region_candidates.py`): D recall 2/2, F recall 2/2 on known
  instances after 3/2 attempts (glyph clusters + large-component path + min-glyphs 2; pocket
  area cap 120k). Accept: gold candidate pools are ~94% harmful (3.5-4.1M px harm vs
  219-272k benefit); best classical rule 52.6k benefit/120.7k harm (D), 0 benefit (F) vs
  bars harm<=5k benefit>=100k. Classifiers (50k-param CandidateNet, 100% synthetic): F
  inverts on gold (top-30 = 26 harmful; the manhwa's system-UI windows are locally identical
  to synthetic "good" pockets with opposite label — structural negative, attempt 1 + stated
  infeasibility); D transfers directionally but the full 3-attempt ladder (plain synth; +
  synthetic UI-window negatives; + 690-width scale alignment) tops out at 2.5k captured
  benefit — 40x short. v7 remains production; no v8 file. Gutter/BandNet stays closed. The
  semantic floor now confirmed at region scale in both rule and classifier paradigms.
- **v21 spiky-cloud + border-residue round: spiky ACTION mechanized and verified (GUI
  track), auto-scope honest negative; border residue = coverage gap, FIXED — v8 SHIPS
  (= v7 + A'3 sandwich-gated [230,256) border sweep)** (2026-08-08). Spiky: manual recipe
  decoded from `spiky-clauds/` and verified exactly (tol-200 = min-channel>=55, 100.00% of
  etalon deletions; crop at (26,77363) in 019.png byte-identical); `clean_spiky_region`
  bbox action: 19,268 under-del -> ~300, over 0 above the frame line; automatic scope
  failed a 3-attempt ladder (unscoped +19.8M over on ch001 GT; enclosure-annulus flood
  safe but 31% recovery; non-connective solves crop but false-positives include a face) —
  object-level semantics, same wall as v20 D/F. Border residue: v5->v6->v7 ladder on new
  full-chapter manual etalons shows a monotone trail (no regression); [250,256)+[230,240)
  bands near frames were never covered (26.7k/53.7k px on 001/002); A'3 (sandwich gate:
  <=3px from both ink and deletion) is net-positive on 001/002 AND the 035 holdout, added
  over-del is 90% <=4px frame-hugging slivers (1 px GT-AA); battery: all gold parts
  improve, fit improves, failing-5 untouched, guard flags visually verified. v8 default =
  v7 + A'3; steps='' = v7-exact; spiky auto variants retained as documented negatives.
- **v22 new-gold PSD round: black-track method DECODED (mask-hard/soft = dark-side
  candidate masks, T=32/64 + dilate-4, IoU 89.6/92.7%); dark backdrop measured as the
  entire remaining gap (under-dark 25-44% of page on 7/10 parts, 033-2/3/4 measured for
  the first time); full-auto scope = honest negative after a 3-attempt ladder; flatness
  rule (dark<=64 & std21<=2.0 & margin/deleted-connected & ~protected) SHIPS OPT-IN as v9
  `steps='D'`** (2026-08-08). Opt-in gains on backdrop parts: 28.6->7.3, 36.6->6.8,
  38.5->4.7, 44.0->11.9% full-error; dark-art parts would regress (001-2 +6.0pp) — the
  human keeps the per-part decision, exactly as the PSD evidence shows the user always
  did (their own candidates: 033-2 applied at 100% dark recall, 001-2 rejected at 1%
  precision). v9 default == v8 byte-identical (battery deltas +0.0000pp, all 10 parts).
  Battery expanded to 10 gold parts with restated bars (failing-5 unchanged-vs-documented;
  guard within v21-adjudicated sliver counts) — PASS. PSDs used for GT/method extraction
  only, no training on real data.
- **v23 spiky-cloud SCOPE round: SOLVED — two-signal cascade (radial run count >= 50 on
  the elliptical rim + interior glyphs >= 5 at ink<100) hits precision 1.0 everywhere
  measured: 13/13 TP + 0/85 FP on 001/002, 1/1 cloud + 0/20 FP on the unseen 019 slab,
  5/5 verified-genuine sites on 007/008; spectral periodicity reported as a no-signal
  honest negative** (2026-08-08). Auto path reproduces the v21 manual-bbox action to
  within 4 px. New real finding: the action's bbox margin blanked a ~60px strip of
  unprotected panel art (the true identity of the 019-2 "below-frame" over-deletion);
  fixed by panel-line flood clipping (`clean_spiky_region_clipped`). Ships OPT-IN as v10
  `steps='S'` (default == v9): the residual fit-page delta (+0.36pp) is between-ray soup
  kept by 005-1's pre-recipe GT but deleted by the current verified recipe — a GT-era
  conflict, not content loss; default-on deferred until that GT is refreshed rather than
  restating a bar against disputed ground truth.
- **v24 frame rounds: metric v1 retracted (source-thinness artifact; pipeline deletes zero
  gray<=100 px) -> AA-nibble metric adopted as the standing spiky-action bar; frame-junction
  nibble confirmed (S added 1.6k/0.8k/0.06k/0.2k px on 002/007/008/019) and fixed by a ±3px
  frame-run protection band (97-100% removed); frame-interior damage (25,208 px inside
  `_protected_interiors`, A' 11.2k + earlier 14.0k, spiky action zero) restored by enforcing
  the existing protection as final step Q — v11 SHIPS (default = v10 + Q; steps='S' =
  frame-guarded action)** (2026-08-09). Legit in-interior deletion measured at 3,122 px
  total (one empty thin box, 002-2, white-on-white restore); battery PASS with every gold
  part improving or holding and fit page -0.0144pp. Sealed-interior misclassification
  deferred pending user crops.
- **v25 DIAGNOSIS (no fixes): the PSD diagnostic set (12 instances) traces 002_5/002_6,
  the v24 step-Q full-page regression, and "Issue 3" to ONE root cause —
  `_protected_interiors` over-extension at full-page scale** (2026-08-09). Evidence
  chain, all vs fresh PSD-etalon alpha GT: (1) both 002 sites WERE accepted by the v23
  cascade (not a recall miss); (2) the under-deleted px are pure white (p50=255), not
  ray ink, and only 0-5% lie in the action's sealed spiky interior — hypothesis of
  sealed-interior misclassification REJECTED by measurement; (3) stage-by-stage: the raw
  action INCREASES under-deletion (+39k/+41k px at the two sites) via its final
  `&= ~(interior|prot)` restore line, and the restored px are 100%/100% inside
  `_protected_interiors` — the full-page prot mask covers inter-panel background sealed
  by full-width frames + page edges, and the action (since v8) wholesale-restores base
  deletions there; (4) step Q (v24) is the SAME over-extension applied page-wide (002
  under 10.4->21.8%, 019 deletion 39.4->18.0%); part-level batteries missed both because
  50k crop edges unseal the geometry (NEW STANDING LESSON: interior/protection changes
  must be validated at full-page scale). v23's site harness was additionally blind to
  action UN-deletions (measured only added px). 019_3 = occlusion-gap leak: the cloud
  body hides the panel frame line, so the v10 clip flood pours into the panel (over-del
  5.2%); v11's Issue-1 band applied but covers a different geometry — and the
  "presumed-normal" set is NOT clean: 019_0 (11.1% over), 019_6 (10.4%), 019_2 (4.9%),
  019_4 (2.8%) share the same leak class; 019_1/5/7/8/9 are clean (over <=0.3%).
  PROPOSED (not implemented): (a) fix `_protected_interiors` scope directly — an
  interior must be bounded by a closed frame CONTOUR, not merely border-disconnected
  (e.g., require an enclosing ink rectangle/contour, or cap interior area, or require
  interior not to touch full-page-width frames on both sides); re-derive Q and the
  action restore from the corrected mask; (b) occlusion leak — bridge collinear frame-run
  segments across a spiky bbox before the clip flood; (c) re-validate everything at BOTH
  part and full-page scale + the diagnostic set as a regression suite. v11 default
  remains UNSAFE for full pages (use v10 or v11 steps='' equivalents until fixed).
- **v26 fix round: `_protected_interiors_v2` (closed-contour ownership test) fixes the
  v25 root cause; occlusion bridging + ellipse scope + saturation gate close most of the
  leak class; v12 SHIPS as candidate (suite PASS, battery PASS incl. new full-page gate,
  chapter GT better than v10-S on both axes); v11 DEPRECATED/UNSAFE (kept for history);
  v10 stays production default until v12 is confirmed** (2026-08-09). Full config
  comparison in logs/v26_suite_all.log: A alone fixes 002 (21.5->1.8% under) but exposes
  3 previously-shielded action leaks; B alone fixes straight-line occlusions only; final
  A+B+E+S: clean sites <= 0.14% over, overlap residual 1.9-3.6% on 019_0/3/6 (achromatic
  bright art in the overlap zone — locally identical to soup; strict <= 0.3% bar unmet
  there, documented open). Standing infra: `v26_fullpage_suite.py` (12-instance PSD
  suite, required gate) + `v26_battery.py`.
- **v27 step 1-2 (reconciliation, before any new fix): class C (interior text-glyph
  dropout) does NOT reproduce on the current v12 (A+B+E+S) config — 0 dropout px measured
  across all 12 instances, using the EXACT site-bbox geometry the action computes (not an
  approximated crop window)** (2026-08-09). Root-cause check on the hypothesized
  mechanism: `_enclosed` (replicate_pipeline_v8.py) marks ANY passable region not
  border-connected as "interior," regardless of whether it is flood-connected to the
  main bubble interior — so an isolated pocket like a letter's counter (e.g. О/Ф) already
  forms its own tiny non-border-connected component and is already protected today. The
  user's proposed mechanism ("connectivity-based flood-fill, single seed, continuous-ON,
  can't reach isolated pockets") does not match this code's actual algorithm (closer to
  "NOT reachable from the border" than "reachable from one seed"). Visual confirmation:
  zoomed renders of 002_5 (dense paragraph) and 019_1 (text-heavy) show zero artifacts on
  letters; the only px flagged are known ray-tip under-deletion, unrelated to glyphs.
  **Class C: no fix needed, closing as a non-issue** (not a 3-attempt stopping-rule
  exhaustion — the defect was never observed to begin). Reconciled the user's 019_2 visual
  flag the same way: fresh v12 measurement shows 019_2 at 0.006% over (clean) with a clean
  render — the leakage the user saw was almost certainly from the earlier v11 crops sent
  before the v26 fix landed, not the current state. **019_3 (2.5% over) and 019_6 (1.9%
  over) DO show real residual leakage** in fresh renders — small red bleed at ray/frame
  crossings into panel art below the line, confirming class B is still open on those two
  instances specifically (019_0's 3.6% over not yet re-inspected visually this round).
  Proceeding to class B fix attempts next.
- **v27 class B (residual silhouette leakage): 3 CONSECUTIVE FAILURES, STOPPING per the
  rule — root-cause diagnosis + attempt log** (2026-08-09). Diagnosis first (render with
  raw-run/bridged-run/action-added overlay on 019_3/019_6): `bridge_added=0` at both sites
  — bridging never even triggers, because the panel below each of these two clouds has NO
  detectable frame run near the boundary at all (the "line" is drawn as vertical
  stripe/motion-line art, not a solid horizontal bar) — so the clip flood finds nothing to
  stop it until it reaches a real line much further down the panel. This is a distinct
  geometry from the v26 occlusion-bridging case (a hidden-but-present line) — here there
  is no line to bridge to nearby.
  Attempt B1 (ELLIPSE_MAX 1.45->1.20->1.05->0.95, single variable): monotonically trades
  over-del for under-del at a ratio far worse than break-even, and the under-del cost
  lands on EVERY instance including all previously-clean ones (e.g. 002_5 1.96%->11.46%,
  019_5 0.22%->5.48% at 0.95) for only a partial over-del win on the 2 target sites (019_3
  2.47%->0.09%, 019_6 1.88%->0.01% at 0.95). Net-negative at every tested value. FAIL.
  Attempt B2 (RUN_KERNEL_W 101->61->31, single variable, ELLIPSE_MAX reverted to 1.45):
  019_3 barely moves (2.468->2.397->2.256), **019_6 is EXACTLY unchanged at 1.879% at
  every kernel width** (proving the leak there has zero relationship to run-detection
  sensitivity), and under-del cost still appears on every instance at width 31. FAIL.
  Attempt B3 (RING_PX distance-to-ink gate 80->50->30, single variable, RUN_KERNEL_W
  reverted to 101): near-total no-op — 019_3 2.468->2.465, 019_6 1.879->1.827 even at the
  aggressive 30px value, with negligible effect elsewhere. The leak pixels are apparently
  immediately adjacent to real ink already (soup always is), so ink-proximity cannot
  discriminate them from genuine soup. FAIL.
  **STOPPING class B at 3 consecutive failures, per the pre-stated rule.** Root cause:
  whether a given achromatic, ink-adjacent, in-ellipse pixel below an occluded/absent
  frame line is "soup" or "panel art" is not resolvable by any of the three geometric
  proxies tested (region-shrink, run-topology, ink-distance) — the same semantic-locality
  wall this project has hit repeatedly (v20 D/F, v22 dark-backdrop). Residual: 019_0
  3.644%, 019_3 2.468%, 019_6 1.879% over-deletion (all improved 3-6x from the v25
  pre-fix baseline of 11.2/5.3/10.5%, but not closed). All three attempts' module flags
  reverted to their pre-attempt (shipped v26) values; file verified unchanged by rerunning
  the full battery+suite with a cleared cache — byte-identical PASS to the v26 state
  (logs/v27_final_battery.log).
- **v27 class A (frame-junction damage) re-verified on the current v12 (A+B+E+S)
  config: CLEAN, no attempts needed** (2026-08-09). The AA-nibble metric (the v24-adopted
  standing bar, not the retracted v1 source-thinness metric) measured 0 or negative added
  nibble on all four reference regions vs the base pipeline: 002 -4px, 007/008/019slab
  0px — all far under the v11 reference of 42/5/0/0. Verified with fresh numbers this
  round, not assumed from v26.
- **v27 FINAL STATUS: v12 (A+B+E+S, unchanged from v26 — no code changes shipped this
  round) remains the candidate.** Session budget did not permit exhausting class B with a
  4th novel mechanism beyond the pre-stated 3-attempt stop; this is the honest stopping
  point per the rule, not a premature cutoff. Class C closed as a non-issue (mechanism
  doesn't exist in this codebase). Class A confirmed clean. Class B stopped at 3 failures
  with the residual documented above. Full battery+suite PASS confirmed on a fresh cache.
  v10 remains production default; v12 remains the candidate pending the user's decision
  on whether the class-B residual (3 instances, 1.9-3.6% over-del, all better than v11)
  is acceptable to ship as-is.

## Repository cleanup (2026-08-10 14:10 EEST)

User-directed 10-point cleanup after both tracks concluded (ML: Recipe A/semantic floor
2026-08-06; classical spiky-cloud: v27 2026-08-09). Commits 7.13.0-7.13.3.

- **CascadePSP removed entirely** (checkpoints `data/models/cascadepsp-*` ~4.3GB deleted;
  `probe/train/export_cascadepsp*`, `ensemble_refine`, `train/probe_refine_head` modules
  and the `--cascadepsp-refine` ml_cleaner path removed). Reasons: results did not justify
  keeping it, and its base weights' upstream training-data license provenance was never
  auditable — incompatible with the MIT-clean policy. The evaluation record in
  `docs/ml_strategy_history.md` remains authoritative.
- **ToonOut probe removed** (`probe_toonout.py`, archive outputs): poor probe results.
- **Dead-code sweep of `src/research/`**: all closed-negative-mission modules deleted
  (halo refiner set, contour/Deep-Snake attempt 7, CRF attempt 8, instance bubble/SFX
  pipelines, task_queue runner). Adopted mechanisms (reclaim_patchy_deletion,
  d1_region_vote, eval_gen6_checkpoint, repair-frames family) and
  rejected-but-doc-referenced modules (semantic_region_vote, panel_border_completion,
  reclaim_black_backdrop) stay. Everything deleted is recoverable from git history.
- **Failed-line disk reclaim**: `.tmp/checkpoints/{stage2,stage3,contour_deform_smoke,
  crf_refine_*,instance_*_smoke}` and `.tmp/datasets/{stage3_sfx_2k,b2_bubbles_2k_prestage}`
  deleted (~3.6GB; datasets regenerable from tracked generators). All stage1*/blackbg
  checkpoints (incl. unadopted v4/v5) and datasets kept.
- **`.tmp/archive` moved to repo-root `archive/`** (gitignored). All scattered `*.log`
  files consolidated into `archive/logs/` with source-prefixed names; papers library at
  `archive/articles/`; closed private cascadepsp notes at `archive/notes-closed/`;
  superseded one-off scripts at `archive/scripts-manual/superseded/`.
- **Spiky pipeline promoted to `src/spiky/` (git-tracked)**: replicate_pipeline v2-v12
  import closure + band_classifier/leak_detector deps, v12 config wrappers, v26
  battery/full-page suite, v27 reconcile, PSD ground-truth extractors (psd_extract,
  psd_extract_gold). Data/caches stay in `.tmp/` (`scripts-manual/{gold_extracted,
  suite_cache,spiky-clauds}`); stale pre-move data paths (`.tmp/minmax`,
  `.tmp/spiky-clouds-diagnostics-psd` → `.tmp/debug/...`) fixed in the process.
- **`.tmp/diagnostics` grouped into per-mission subfolders** (halo/, blackbg_darkpanel/,
  recipe_a/, patchy/, sfx/, spiky/, border_probes/, misc/) + README index. Paths cited in
  docs/notes before 2026-08-10 are the old flat ones — same files, now inside subfolders.
- Stray `regression_summary.json` (root + .tmp) and `.tmp/task_queue/` deleted.
- **Verification**: all src/spiky + kept research modules import/compile clean;
  `ml_cleaner --help` and a production smoke run (10.0-baseline + --reclaim-islands)
  pass; v26 battery + 12-instance suite PASS byte-identically from the new location;
  psd_extract 0-mismatch on the 019_5 x-offset PSD. Disk: repo 81GB → 74GB.

## src/spiky consolidation (2026-08-10 14:45 EEST)

Follow-up to the 7.13.2 promotion, per user direction: the per-version file chain
(`replicate_pipeline_v2/5/6/7/8/9/10/11/12.py` + five `v12_cfg*.py` flag wrappers) is
merged into a single `src/spiky/pipeline.py` — version history belongs in the git
commits, not as parallel files. Sections keep their vN provenance headers; entry points:
`clean_page` (v12 candidate, ABES defaults), `clean_page_v10` (production),
`clean_page_v7` (battery white-track reference), `apply_config(letters)` (replaces the
cfg wrapper files), `clean_spiky_region` (manual/GUI action). v26_fullpage_suite's
importlib-by-module-name mechanism became config-letter based; cache keys unchanged.

**Equivalence verified BEFORE deletion** (old chain vs pipeline.py, np.array_equal):
v7 on gold parts 001-1/033-2 EQUAL; v12 ABES "QS" on full 002 and 019 EQUAL old-vs-new
AND vs the pre-existing suite_cache masks; v10 "S" on full 019 EQUAL. Full v26 battery +
12-instance suite re-run from the consolidated module: PASS, numbers identical.
Commit 7.13.4.

## Eval-data path neutralization (2026-08-10 15:25 EEST)

Post-audit cosmetic fix (the MIT audit itself found no violations): tracked src/spiky
files no longer embed the eval manhwa's title in path strings. They now read through
neutral local symlink aliases under gitignored `.tmp/eval/`:
`002.png` -> `.tmp/debug/minmax/other/002.png`; `019.png` and `merged` -> the merged
eval-chapter folder under `.tmp/saved/materials/Merged/`. Recreate with `ln -sfn` if
`.tmp` is rebuilt. The unpushed commits 7.13.2-7.13.4 were history-rewritten to carry
the neutral form too, so no pushable blob names the work. Battery re-verified PASS
through the symlinks. Commit 7.13.5.

## Generation 8 mission start: modular detector framework (2026-08-10 18:11 EEST)

New parallel architecture effort on branch `testing` (normal branch off `main` @
`a13c777`, shared history — verified `git merge-base --is-ancestor main testing`).
Standing principles for the whole mission: classical rules/geometry only (no ML,
including fallbacks — deliberate architectural choice); when uncertain, DELETE
background over PRESERVE frame content (baked into decision boundaries); never train
on real manhwa (eval against real chapters/gold PSDs fine); one variable per attempt,
commit every attempt; classifiers live in `src/classifiers/`; per-classifier merge to
`main` only with explicit user confirmation. Phases: 1 extract background classifier
(mechanical, byte-identical), 2 geometric frame classifier (Hough + missing-side
extrapolation, targets the v27 class-B occlusion residual 019_0/3/6), 3 regression
suite for the existing cloud classifier, 4 detector framework with pluggable geometry
profiles (spiky_cloud ported from v23 cascade, sfx_glyph new, regular_cloud from phase-3
findings). Fallback protocol: 3 attempts → root-cause diagnosis → new hypothesis
family → 3 more → honest negative, flag for GUI/manual track, move on.

Setup also recorded here: `.gitignore` hardened with global `*.npy`/`*.npz`/`*.pth`
(no tracked instances; `data/models/*.pt` stays tracked). Precondition executed before
setup: `notes/` and `archive/` had vanished from disk (13:27-16:25 today, cause
unknown, never git-tracked) — fully restored from timeshift snapshot
`2026-08-10_13-27-45` with the day's reorg re-applied (CASCADEPSP/toonout deletions,
log consolidation, superseded scripts, notes-closed); full battery re-run after
restore: PASS with standing numbers.

## Gen-8 phase 1: background classifier extracted (2026-08-10 18:26 EEST)

`src/classifiers/background.py` now holds the validated connected-component/
reachability primitives, MOVED verbatim from `src/spiky/pipeline.py` (no reimplementation):
`enclosed`/`flood` (v8 exact 4-conn primitives), `protected_interiors` (v6 hole
detection), `protected_interiors_v2` (v12 Fix A closed-contour ownership test, the
full-page-context fix) + `PROT_DOMINANCE`, plus a new `classify_background(page_rgb)`
convenience entry (protection scoped to ownership-passing interiors only —
delete-over-preserve bias documented in the module). pipeline.py imports them under
the historical underscore names (zero call-site changes); v27_reconcile re-pointed to
the module directly. Siblings deliberately not merged: `ml_cleaner.
repair_frame_interiors`, `style_analysis.extract_enclosed_holes` (different lineages).

Verified byte-identical (this was a refactor; any diff = bug): np.array_equal fresh
post-extraction vs pre-change references — v7 on gold 001-1 (pre-saved mask), v12-ABES
"QS" on full 019 AND 002 vs suite_cache, v10-S on full 019 vs suite_cache: all EQUAL.
Full battery + 12-instance suite: PASS, numbers identical. Commit 8.1.2.

## Gen-8 phase 2: frame classifier — hypothesis family 1 (line detection) 3 failures, root cause identified (2026-08-10 18:32 EEST)

`src/classifiers/frame.py` built with the full attempt ladder; measured against the
class-B leak sites (019_0/3/6) on full chapter 019:
- **A1 (probabilistic Hough + rectangular grouping): FAIL.** Page-wide h=202/v=6154
  lines, but 0 h-lines inside the 019_0/019_3 windows — Hough misses even lines the
  plain erode/dilate morphology finds (accumulator/sampling dilution on a 153k-row page
  dense with art ink). Evidence: local bridged-run rows exist at 1714-1716/1850-1851
  (019_0) with no corresponding Hough line at any coverage.
- **A2 (missing-side extrapolation on A1's inventory): FAIL.** 100 panels grouped
  page-wide, but extrapolation cannot conjure sides whose horizontals A1 never
  detected: 0 new in-window frame rows at 019_0/019_3; the single row at 019_6 (32024)
  duplicates local knowledge.
- **A3 (page-wide morphological long-run inventory + collinear occlusion bridging,
  both-flanks >= 100px evidence rule): FAIL on the success criterion.** The inventory
  itself is good (1583 bridged h-lines in 0.4s, incl. a full-width line at 1715
  spanning 019_0's window) — but it supplies ZERO barrier rows the window-local Fix B
  signal doesn't already have, at all three sites.

**Root cause (family-level):** the v27 class-B residual is NOT frame-line occlusion.
Every frame line that exists at those sites is already found locally; the leakage
traverses regions genuinely devoid of line evidence at any scale (irregular/organic
panel boundaries). The phase premise ("occlusion needs global extrapolation") is
measured false for these instances. Per the fallback protocol: line-detection family
STOPPED at 3; second hypothesis family = AREA evidence — use the detected panel
RECTANGLES (the classifier's grouping output) as interior barriers for the spiky
action's scope, pending a coverage diagnostic (do detected rects cover the leaked px?).

The module itself stays: `classify_frames` (A3 configuration), `detect_lines` (A1,
kept for the record), `detect_lines_morph`, `bridge_collinear`,
`extrapolate_missing_sides` — a working page-scale frame-line/panel inventory for
framework use regardless of the class-B outcome. Commit 8.2.1.

## Gen-8 phase 2 CLOSED: honest negative on class-B via frame geometry (2026-08-10 18:33 EEST)

Hypothesis family 2 (panel-RECT interiors as area barriers for the spiky action)
died at its pre-attempt coverage diagnostic — no integration attempt was warranted:
detected rect interiors cover 0.1%/0.0%/0.0% of the actual leaked (over-deleted) px at
019_0/019_3/019_6, while containing 100%/23%/34% of the CORRECT deletions (791 rects
from the greedy grouping over-cover background) — as a barrier it would block genuine
deletion massively while missing the leak entirely. The leaked px lie in
irregular/organic panel regions that produce neither line nor rectangle evidence.

**Phase-2 verdict:** two independent classical-geometry hypothesis families (global
line detection ×3 attempts; area/rect evidence, closed by diagnostic) cannot address
the v27 class-B residual. Combined with v27's own three failed local attempts, class B
(019_0/3/6, 1.9-3.6% over-deletion) is now flagged for the deferred GUI/manual track —
per protocol, no third family without explicit user go-ahead. Existing v12 numbers
unchanged (battery/suite untouched this phase — no pipeline modification was ever made;
all measurement used cached masks + read-only probes).

Module status: `src/classifiers/frame.py` ships as a page-scale LINE inventory
(`detect_lines_morph` + `bridge_collinear`, deterministic, 0.4s/153k-row page,
validated against the local signal) for framework use. `classify_frames`' panel-rect
grouping is NOT validated for standalone use (over-generative, see diagnostic above) —
documented in the module. Commit 8.2.2.

## Gen-8 phase 3: cloud-classifier regression baseline established (2026-08-10 18:37 EEST)

Classifier identified: `style_analysis.extract_enclosed_holes` + `classify_and_measure`
(the 5-family taxonomy). The user-reported defects map exactly to the documented record
(notes/style_analysis_findings.md): (a) recall gap = limitation 4 (dark-scene flood-fill
break, bounded not fixed) + limitation 2 residual; (b) frame-as-cloud FP = limitation 2
(routing); (c) the "existing untested fix" = the Revision-2 pair (width-only frame
routing + text-plausibility interior filter), shipped 2026-07-26 on visual spot-checks
only. New suite: `src/classifiers/tests/cloud_suite.py` — 92 user-curated clauds crops
(set A, padded-canvas harness documented in the module), the 12 spiky PSD instances
(set B), 20 synthetic frame-only pages (set C). Coverage gaps stated in the module
docstring (presence/family GT only, dark-scene coverage limited to what ch2 crops carry).

**Baseline (current code, first measured numbers):**
- Set A recall 68/92 (73.9%) — all 24 misses are "no candidate hole at all", not
  filter rejections. Adjudication caveat: inspected misses include crop-TRUNCATED
  outlines (open ink loop at the crop edge → no enclosed hole by construction), so
  24 is an upper bound mixing harness artifacts with genuine detection gaps; per-crop
  adjudication is phase-4 regular_cloud input, not assumed here.
- Set B 12/12 (100%) — every spiky instance detected, all classified 'thorn'
  (corroborates limitation 6's thorn/spiky boundary fuzz; detection solid, family fuzzy).
- Set C: 4/20 frame-only pages produce bubble-family FPs (all 'rectangle' — panel
  rects entering the bubble taxonomy). Defect (b) is real and now has a number.
- Defect (c) verdict on this evidence: the Revision-2 text-plausibility fix holds on
  the measured set — text-heavy boxes/bubbles detect fine; zero observed
  rejected-by-filter misses. Not a full clearance (no labeled Revision-1 art-FP cases
  in the suite), but the fix is no longer purely visually-verified.

This table is the baseline phase 4's regular_cloud profile must beat. Commit 8.3.1.

## Gen-8 phase 4 (part 1): detector framework + spiky_cloud profile ported (2026-08-10 18:40 EEST)

`src/classifiers/detector_framework.py`: one detection loop, pluggable Profile =
candidate generator + independent geometric Signals, AND-voting (the v23-validated
cascade structure), `detect(page, profile, explain=)` with per-signal value reporting
(v20/v23 trade-visibility standard). `src/classifiers/profiles/spiky_cloud.py`: the
v23 cascade RESTRUCTURED into that shape — candidate generation verbatim, both signals
via the same `_rim_runs_and_glyphs`, every constant IMPORTED from pipeline.py (no
number duplication). **Equivalence gate PASS**: `detect(page, PROFILE)` returns
site lists IDENTICAL to `pipeline.find_spiky_sites` on full chapters 002 (10 sites)
and 019 (27 sites). pipeline.py remains the production caller — re-pointing it at the
profile is a separate explicit decision per the per-classifier merge rule. Commit 8.4.1.

## Gen-8 phase 4 (part 2): regular_cloud profile adopted at FP 4->1, zero recall cost; sfx_glyph blocked on labeled data (2026-08-10 18:50 EEST)

`src/classifiers/profiles/regular_cloud.py` — existing classifier as candidate
generator + framework signals. Attempt ladder (full log in the module docstring, one
variable each): A1 frame-line alignment, fixed 3px tolerance — counted failure
(0 FP fixed; root cause: line pos = stroke center vs hole edge at stroke inner edge,
19-38px offsets on thick synthetic strokes). A2 thickness-aware alignment
(`Line.thick` added to the frame module) — SUCCESS, FP pages 4->2, recall preserved,
counter reset. A3 stroke-thickness signal — counted failure (regress-elsewhere: FP
2->1 but set A 68->62; ring measurement conflates enclosing stroke with adjacent
art/dark-scene ink — all 7 false rejections were thorn/other, all true FPs
'rectangle'). A4 class-scoped thickness (rectangle-only) — SUCCESS: **set C FP pages
4 -> 1, sets A (68/92) and B (12/12) fully preserved.** Adopted profile =
thickness-aware frame-line alignment + rectangle-scoped stroke-thickness gate.
Residual: 1 FP page (000016, a page-edge box whose ring measurement is diluted at the
canvas boundary) — documented, next single-variable candidate is edge-aware ring
handling; ladder remains open (2 successes, counter at 0).

`sfx_glyph` profile: BLOCKED, not attempted — no labeled SFX-glyph ground truth
exists anywhere in the repo (the deleted find_sfx_instances work left only aggregate
precision numbers; .tmp/diagnostics/sfx/ holds unlabeled candidate renders). Building
a suite-less profile would fabricate confidence. CONCRETE ASK: a small user-curated
SFX crop set (like the clauds-and-ui crops) unblocks it immediately.

Battery bookend: full battery + 12-instance suite re-run at mission end — PASS,
numbers identical to mission start (pipeline untouched since 8.1.2's byte-identical
extraction; classifiers are additive modules). Commit 8.5.1.

## Gen-8 sfx_glyph unblock, step 1: 6 SFX reference PSDs decoded — both threshold passes solved pixel-exact, per-object Expand measured (2026-08-11 15:42 EEST)

The concrete ask from 8.5.1 was answered: 6 reference PSDs + written recipe
(`.tmp/scripts-manual/SFX/`, "SFX Pipeline" prose) documenting the user's manual
two-layer SFX process. Same discipline as v12/v21/v25: PSD layer data is ground truth
over the prose. `src/classifiers/tests/sfx_decode.py` decodes all 6 (GT keep/delete via
the validated psd_extract `.composite()` path, canvas-scoped — each PSD is a crop
window into a full page positioned at a large negative layer offset).

**Threshold passes SOLVED, pixel-exact on all 6 files (0 mismatch px anywhere):**
- Aggressive pass (`img-clone`, all 6 files): effective predicate **G >= 33** —
  match 100.000%. Exactly the recipe's Levels(32,1,33)->Threshold(140) composition:
  per-channel Levels binarizes at 32.5, Threshold 140/255=0.549 needs only the green
  channel's Rec601 weight (0.587).
- Preservation pass (`img-copy`, present in 005.psd only): effective predicate
  **min(R,G,B) >= 50** — match 100.000%. Exactly Levels(49,1,50)->Threshold(230):
  230/255=0.902 requires ALL channels white. It rescues saturated mid-tone color
  (005's blue-gradient glyphs): 18/18 rescued diff components are 100% kept in GT.
  The two passes differ by predicate STRUCTURE (G-only vs min-channel), not just value.
- Both cutoffs constant wherever present; no third value found in any file.

**Per-object Expand measured** (annulus keep-profile E_prof + nearest-delete E_del,
cross-checked, neighboring-object ownership partition): clean isolated/gutter SFX show
E ~2-3px at stroke width ~1.9px and E ~4-5px at stroke width 2.7-5.5px — matching the
prose's "2px small / 4px+ large". Best simple rule on the ~12 measurable objects:
two-level E = 2 if stroke_w < ~2.5px else 4 (fits within ~1px; sample too small to
prefer a continuous fit; delete-over-preserve favors the smaller side).

**Structure confirmed visually** (overlays in `.tmp/sfx_decode/`): whole-frame-interior
keep (case 2/3 SFX cross panel borders seamlessly); crisp ~4px halos on gutter glyphs;
004(2)/004(3) additionally keep gutter SPEECH BUBBLES wholesale (regular_cloud
territory present in SFX GT — flagged for the prototype). Honest GT observation:
in 004(1) (pass-1-only file) a light-red stroke segment of a glyph was thresholded
away and IS deleted in the user's own GT — the exact defect pass 2 exists for,
accepted manually there.

Case coverage: case 1 (isolated gutter SFX) and case 2 (crossing frame edge) well
represented; case 3 (fully inside frame) present in 004(2)/004(3)/004; no file is
case-3-only. Decoded arrays cached to `.tmp/sfx_decode/export/*.npz` for downstream
steps. Commit 8.6.1.

## Gen-8 sfx_glyph step 3: profile shipped after a 6-attempt ladder (1 counted failure with measured root cause) (2026-08-11 16:02 EEST)

`src/classifiers/tests/sfx_suite.py` (labeling harness: manual frame-rect reference
annotations anchored to the validated line inventory; auto-labels frame_line /
bubble_part / sfx; per-candidate geometry features) + `src/classifiers/profiles/
sfx_glyph.py` (full attempt log in the module docstring).

Evidence-first signal choice: measured feature table showed SFX geometry OVERLAPS
bubble text (both glyphs) and bubble outlines / frame lines (thin uniform curves) --
context separates those, and the ladder proved the context filters belong in the
COMPOSITION, not the profile: gutter flood is fragile (leaks through border runs
broken by bubbles/steam, over-seals pockets; 4 measured misses) and ring-enclosure
cannot tell sealed gutter pockets from bubble interiors (2 measured misses at
ring=1.0). Both filters dropped (A2, A3 -- recall 16->22); over-admission is
pixel-harmless because frame-keep/bubble-keep cover those regions downstream.

Line-structure FPs killed in three steps: A4 pixel-coverage-under-inventory-lines
(synth 14->0 but recall -4: COUNTED FAILURE -- inventory art-mass entries with
thick=100-300 drawn at measured thickness swallow real SFX), A5 barrier thickness cap
30px (recall recovered), A6 boundary-concentration signal (border bands/rect outlines
0.93-1.00 vs glyph strokes <=0.74; synth FP pages -> 0/20).

Final: **recall 20/22** (2 permanent misses are border+steam merged components, kept
via frame interior in composition -- pixel-harmless), **harmful extras 0** (no
detection sits on GT-deleted ink anywhere in the 6 refs), **synth FP pages 0/20**,
chapters 002/019 run in 5-8 s (~1500-1750 stroke-structure detections; visual sample
audit: real SFX + bubble text + in-frame art strokes, both latter classes harmless).

Also fixed here: id()-keyed page caches held no reference to the page, so a freed
array's recycled id could serve a stale context (measured: 004_4 evaluated with
another file's lines). sfx_glyph now holds the page reference; the same latent hazard
exists in spiky_cloud/regular_cloud caches -- fix + gate re-verification queued as a
separate commit. Commit 8.7.1.

## Gen-8 integrity fix: id-recycle cache hazard in spiky_cloud/regular_cloud; 8.5.1 set-A number corrected 68 -> 67 (2026-08-11 16:04 EEST)

The stale-context hazard found during the sfx_glyph eval (8.7.1) also lived in the
sibling profiles: caches keyed by id(page) without holding a reference to the page,
so a freed array's recycled id could serve another page's lines/classes/measurements.
Both fixed (page pinned while cached; regular_cloud additionally purges recycled-id
class entries).

Gates re-run after the fix:
- spiky_cloud equivalence: detect == pipeline.find_spiky_sites, IDENTICAL on 002
  (10 sites) and 019 (27 sites) -- the 8.4.1 result stands.
- regular_cloud suite: B 12/12 and C 1/20 (000016) unchanged; **set A corrects
  68/92 -> 67/92**. The one delta (ch3/clauds/007.png) is a crop whose only
  bubble-family shape is a 7x198 px 'other'-class sliver with frame_align=1.00 --
  a line-hugging artifact, not the crop's actual cloud. Its 8.5.1 "hit" was the
  stale-cache bug masking a legitimate alignment rejection; the corrected number is
  the true profile behavior and the lost hit was spurious to begin with. The 8.5.1
  adoption verdict (C 4->1, recall preserved) stands with A=67 as the honest
  baseline-parity figure (baseline's own 68 included the same spurious sliver).
Commit 8.7.2.

## Gen-8 sfx_glyph step 4: sfx.py composition prototype -- zero frame-loss on all 6 refs, over-delete <= 0.23% (2026-08-11 16:10 EEST)

`src/classifiers/sfx.py`: explicit composition of the three classifiers per the
decoded recipe -- pass-1 aggressive binarize (G >= 33) proposes deletion; subtracted
keeps: (a) FRAME band from border-quality inventory lines (thin <= 20px, span >= 0.4,
non-edge; an axis with < 2 border lines keeps FULL extent -- measured collapse
without this rule: a single-line axis banded to the line's own 2px and deleted the
panel), (b) SFX = profile detections grown by CONNECTIVITY under the pass-2 predicate
(min(RGB) < 50) in a generously padded window, then dilated by the measured two-level
Expand (2px thin / 4px thick strokes), (c) BUBBLES = enclosed pockets >= 3000px
dilated by the same Expand mechanism (the recipe's wand-ON + Expand semantics;
POCKET_MIN raised from 400 after measuring glyph-loop interiors wrongly kept).

Acceptance vs PSD GT (sfx_suite.prototype_eval): over-delete 0.000-0.233%,
over-keep 0.499-1.264% on 5 of 6 files; **HARD zero-frame-content-loss guard PASS:
0 deleted px inside any annotated frame rect on all 6 files.** Iteration log:
initial frame band collapsed on single-border-line axes (fixed, above); pass-2
rescue confined to the detection bbox deleted 005's gutter blue-gradient glyph
whole (fixed via connectivity rescue: 005 over-del 1.414 -> 0.067%).

Known limitations (documented, not hidden): 004_4 over-keeps 27.6% -- its bottom
border is only present inside a thick=169 dark-art inventory entry, so the y-axis
falls back to conservative full extent (the guard holds; the delete bias yields);
sealed gutter pockets are kept as bubbles (wall-material analysis would be needed to
distinguish them -- future ladder); chapter-scale strips need per-panel banding
(phase-2's open rect-grouping problem) -- the prototype targets page/crop scale like
the manual recipe itself. Commit 8.8.1.

## Gen-8 sfx.py residuals, priority reversal round: A1 border extrapolation -- 004(4) over-keep 34% -> 0.5% (2026-08-11 18:33 EEST)

USER PRIORITY CHANGE governing this round: over-delete preferred to under-delete,
under-keep preferred to over-keep (aggressive deletion is manually recoverable,
background mess is not). Explicitly: this applies to keep-vs-delete resolution in
ambiguous background regions ONLY -- the zero-frame-content-loss adversarial guard
stays HARD and unchanged. Dark-background domain stays PAUSED: measured, the 6 refs
are white-domain throughout (GT-deleted background 99.4-100% light per file;
residual 0.26-0.58% dark px are stroke anti-aliasing) -- domain mixing is a
non-issue in this test set; eval now reports white-only AND total denominators.

Metrics note (stated once): binary masks have exactly two independent pixel error
quantities -- FP-delete (we delete, GT keeps) = over-delete = under-keep, and
FN-delete (we keep, GT deletes) = under-delete = over-keep. prototype_eval reports
both, px and %, white-only and total.

**A1 (one variable), frame_keep_mask**: an axis with exactly ONE border-quality
line no longer keeps the full extent. The merged-into-dark-art border is invisible
to the inventory AS A LINE, but the art-mass entry's FAR EDGE is where the mass --
and the border -- ends; the band extrapolates from the border line to the farthest
far-edge among other inventory entries on the panel side (the side holding the
majority of entries). No-anchor safety: with zero other entries the axis keeps full
extent (hard guard outranks delete bias where no evidence exists). Class-B's
"starved" lesson respected: this anchors on measured inventory evidence, not pure
rectangle-geometry prediction (004(4) has only ONE confirmed side, so 3-sides
extrapolation is unanchorable there).

RESULT: 004(4) FN-delete (over-keep) 34.064 -> 0.527% white-only (160,870 -> 2,652
px) at a cost of 20 over-deleted px (0.003%); predicted border row 772 vs
hand-annotated 771. Other 5 refs BIT-IDENTICAL (they never enter the 1-line path).
HARD GUARD: 0 frame-loss px on all 6. Commit 8.9.1.

## Gen-8 sfx.py residuals: residual 2 measured as a NON-ISSUE (B1/B2 honest negatives), C1 rescue seed-fraction gate adopted (2026-08-11 18:37 EEST)

**Residual 2 (sealed-gutter-pocket vs bubble) -- measured, both attempts negative:**
Per-pocket measurement across all 6 refs at POCKET_MIN=3000: every pocket the
bubble-keep touches is 100% GT-KEPT -- the sealed-gutter-pocket population flagged
at 8.8.1 does not exist in the current configuration (it was real at the earlier
POCKET_MIN=400; the 3000 floor already removed it). Consequences, measured:
- B1 (flip ambiguous default to delete, the brief's explicit question): FP-delete
  explodes 0.106 -> 6.862% (004_2) and 0.155 -> 4.295% (004_3) white-only, while
  FN-delete improves by at most 0.022 points. VERDICT: pure error relocation at
  ~500:1 harm ratio, NOT a real improvement under any priority ordering -- bubbles
  are content the manual GT keeps, not background mess. Honest negative, default
  stays keep-all-pockets.
- B2 (wall-material test, threshold 0.30): no sealed pockets exist to catch, and
  the test MISFIRES on a real bubble -- 004_3's (279,664) pocket overlaps the frame
  border (wall_frac 0.41, GT-kept 100%) and would be wrongly deleted. Honest
  negative in the current data; the wall-material idea stays documented for when a
  reference with actual sealed pockets appears. bubble_mode='none'/'wall' remain in
  clean_sfx_region as measurement instrumentation, default 'all'.

**FN attribution** (where the remaining over-keep lives): 0% of GT-deleted px are
pass-1 ink (the formula's floor is clean -- GT never deletes what pass-1 calls
ink); the dominant bucket is sfx_keep over-growth from the unconditional
connectivity rescue grabbing large barely-touching neighbors (frame lines,
adjacent art), plus frame-band margins (8-26%/file) and bubble halos.

**C1 (one variable): rescue seed-fraction gate** -- a touched ink2 component is
admitted only if the detection seed is >= 3% of it (RESCUE_SEED_FRAC; a colored
glyph's dark core is ~9% of its body on 005, so real glyph bodies pass; a grabbed
frame line is <<3% seed). RESULT: aggregate FP-delete 4097 -> 3465 px and
FN-delete 30647 -> 26828 px across the 6 refs -- BOTH independent quantities
improved; per-file: 004_1 improves on both (FP 1157 -> 33 px, FN 4024 -> 2114);
004_2 trades +206 FP for -1163 FN (the priority-preferred direction); 004_3
+286 FP / -43 FN (small, sanctioned direction); 004/004_4 bit-identical.
HARD GUARD: 0 frame-loss px on all 6 throughout -- the priority shift touched only
ambiguous keep-vs-delete resolution, never the guard. Commit 8.9.2.

## Gen-8 sfx.py safety guards (pre-merge blocker closed): zero-line no-op + blank-gutter evidence + band-inversion check (2026-08-11 19:20 EEST)

Visual review found content loss the metrics never saw. Root-cause turned out to be
TWO classes, not one -- the brief's zero-line diagnosis was incomplete and is
corrected here with measurements:

**Attempt 1 -- zero-border-line no-op guard** (the brief's spec): a window with no
border-quality line on either axis returns an all-False delete mask (genuine
no-op; pass-1 never runs). Threshold is EXACTLY zero lines total -- one line still
feeds the A1 far-edge extrapolation. Scope: PROCESSING-WINDOW, because that is the
scope the frame band protects at (lines elsewhere on a page do not protect a
window they don't appear in). Verified: all 66 zero-line windows across chapters
002/004 now delete exactly 0 px; guard fires on NO reference (boundary check
printed per file, 004(4) with exactly 1 line stays on the A1 path).
BUT: the named case 002 y37100 has FOUR qualifying lines (caption-box borders +
building edges) -- its 31.7% art deletion is a SPURIOUS-BAND class, not zero-line.

**Attempt 2 -- blank-gutter evidence guard**: the band must be corroborated by
what it claims -- the outside-band region must actually look like blank gutter
(G >= 200 fraction >= 0.60). Measured separation: refs 0.94-0.96 vs damage classes
0.03-0.28. Closes y37100 (31.7 -> 0.000%) and most cut-panel/dark-scene windows.
Residual found by distribution scan: no clean threshold gap -- 002 y51300-class
damage passes at 0.61 while healthy pink-gutter windows sit at 0.63.

**Attempt 3 -- band-inversion guard**: the y51300 class is structural -- a window
showing two panels cut at its edges yields only their FACING borders as lines, so
the band captures the GUTTER (keeping blank, deleting the cut panels' art).
Directly detectable: inverted bands are BLANKER INSIDE than outside (damage 0.91-
0.99 inside vs 0.61-0.66 outside; refs 0.37-0.55 inside vs 0.94-0.96 outside,
never inverted). inside_blank > outside_blank -> no-op. All 4 measured inversion
windows -> 0.000%.

**GUARD PRIORITY ORDERING (explicit):** hard frame-content-loss guard > safety
no-op guards (zero-line, blank-evidence, inversion) > delete-bias priority >
ambiguous-background-keep. The delete bias applies only where SOME proven frame
protection exists; it was never meant for windows with no protective context.

Verification: 6-ref suite BIT-IDENTICAL to 8.9.2 (incl. hard guard 0 px); A1
region on real chapter unchanged (27.1%); census: 71 of 217 chapter windows
actively clean, 146 are proven-context no-ops; battery + 12-instance suite PASS,
identical to standing bookend.

**Merge-readiness (honest):** sfx.py now degrades to no-op instead of damage on
every measured failure class, at the cost of under-cleaning (some legitimate
gutter stays; safe direction). Recommended application remains explicit opt-in on
panel-complete crops/pages (the manual workflow's own unit): arbitrary fixed-
height windowing still produces thin art slivers at window seams when a cut
panel's remainder passes the guards (seen at 004 y78850's top edge, ~30 rows) --
per-panel banding (the phase-2 open problem) is the real fix for blanket
application. Commit 8.10.1.

## Gen-8 chapter-scale panel segmentation shipped: all 6 refs reproduced, damage classes identified at segmentation level (2026-08-11 19:53 EEST)

`src/classifiers/panel_segmentation.py` -- whole-chapter typed segmentation
(gutter/panel/partial/borderless) removing the fixed-window root cause. Primary
signal: row-blankness band decomposition (whole-chapter, vectorized, ~8 s/chapter
including the line inventory); frame.py border lines then PARTITION bands from
within. 5-attempt ladder, 1 counted failure:

- A1 edge-adjacency classification: COUNTED FAILURE, measured -- floating gutter
  SFX merges into the panel's content band (004(4): border line mid-band, all
  bands borderless) and pale panel interiors split bands (004: panel bottom lost
  75 rows); plus a leading-run merge bug fabricated 2-row bands (fixed).
- A2 line-band reconciliation (new family): lines partition bands; in-band
  intervals classified by own blankness; two-tier run lengths (MIN_GUTTER=80 --
  pale in-panel gaps measure 45-75 rows, real gutters 150+). 4/6 refs exact.
- A3 partial far-edge refinement: REUSES the A1-far-edge evidence (an art-merged
  border ends where its inventory art-mass entry ends): 004(4) partial lands at
  772 vs annotated 771, bottom glyph zone split off as borderless.
- A4 adjacent-gutter border absorption (cap 120 px + blank-continuation test):
  004's thin bottom border, swallowed by the blank run after a pale interior,
  again closes its panel -> (377,912) exact.
- A5 blank-neighbor validation: a border separates content from GUTTER; lines
  deep inside continuous content (caption boxes/building edges on full-bleed
  art) do not. Panel/partial without a gutter or strip end within 300 rows
  demotes to borderless. y37100 -> borderless (identified, not just guarded);
  y51300's brick panel stays 'panel' WITH BOTH BORDERS (the full-chapter-scope
  win: fixed windows only ever saw its facing border).

Validation: all 6 refs reproduce the hand-annotated FRAME_RECTS (4 exact panels,
004(4) = borderless glyph zone + partial at 771+-1 + borderless, 005 exact);
chapters 002/004: 73/71 processing units (gutter-midpoint cuts, panels never cut
mid-body by construction), segmentation maps rendered for review. Chapter-scale
border-quality criteria documented: v-line span is ABSOLUTE (panel-height
scaled), not chapter-fraction (the recurring scale lesson). Commit 8.11.1.

## Gen-8 panel-aware windowing shipped: chapter adversarial 0 px, seams 0, guards silent (2026-08-11 20:04 EEST)

`clean_chapter(rgb)` in sfx.py -- the architectural replacement for fixed-height
windowing. Part-2 ladder:

- First wiring (units processed through clean_sfx_region's own band derivation):
  COUNTED FAILURE, measured -- chapter adversarial found 387k/493k deleted px
  inside panel/partial interiors. Three root causes diagnosed: (1) pseudo-partial
  segments from furniture-edge lines with content on BOTH sides (a real border
  has a blank side: measured 0.94-1.00 vs 0.16-0.23); (2) unit-level A1 bands
  shorter than the true content extent (art rows without 101px dark runs
  contribute no inventory entries); (3) x-band collapse from single v-lines.
  Additionally the 8.10.1 guards, evaluated at unit scope, are DILUTED by the
  unit's legitimate gutter halves (only 5 firings/chapter; damage passed).
- B1 SEGMENTATION-DRIVEN KEEP: clean_sfx_region accepts a precomputed keep_mask
  (None = standalone behavior, unchanged -- 6-ref suite verified BIT-IDENTICAL);
  clean_chapter builds the keep from segmentation directly: panel/partial rects
  + DENSE borderless segments (ink >= 0.15 = full-bleed art islands, kept
  wholesale); sparse borderless (floating glyphs/bubbles) and gutters get gutter
  treatment (pass-1 delete minus SFX/bubble keeps). Head/tail pure-gutter slices
  included after a measured seam check caught them unprocessed.

RESULT (chapters 002/004): **chapter-scale adversarial = 0 deleted px inside any
panel/partial interior** (was 387k/493k on the first wiring); **seam
discontinuities at unit cuts = 0**; deletion coverage 37.7%/37.3% (vs 4.1%/3.6%
in the guard-era chapter run -- the previously guard-suppressed legitimate gutter
is now actually cleaned); ~18 s/chapter. Damage-region verification: y37100
kept wholesale as identified dense-borderless; y51300's panels processed WHOLE
with the inter-panel gutter cleaned; the 177k-px pseudo-partial room scene
(y21891) untouched.

GUARD-REDUNDANCY VERDICT (measured): on the panel-aware path the three 8.10.1
guards are BYPASSED by design (segmentation knowledge supersedes re-derivation;
firings = 0 by construction) -- and the first-wiring measurement showed they are
NOT a sufficient safety net at unit scope anyway (gutter dilution). They remain
active for standalone clean_sfx_region use (defense-in-depth on arbitrary
crops). The root cause (fixed windows cutting panels) is REMOVED, not mitigated:
units are cut at gutter midpoints, panels are whole by construction, and the
keep comes from validated whole-chapter segmentation. Commit 8.11.2.

## Gen-8 finale: merge to main + full orchestration (2026-08-11 20:25 EEST)

Part 1 (user-authorized merge): `testing` merged into `main` -- fast-forward
(testing strictly ahead on shared history; no rewrite, no merge commit needed),
main at 15734d0, 172 commits. On main: full battery + 12-instance suite PASS
identical to the standing bookend; 6-ref SFX suite bit-identical, hard guard 0.
PRODUCTION DEFAULTS UNCHANGED: clean_page_v10 stays the spiky production
default, 10.0-baseline.pt + --reclaim-islands stays the ML path,
pipeline.find_spiky_sites remains the production spiky caller (not re-pointed).
Gen-8 classifiers are importable, additive APIs only.

Part 2 (8.12.1): `clean_chapter_full` -- verified spiky_cloud/regular_cloud were
NOT yet composed (standalone only); wired by composition, not new mechanism:
panel-aware clean_chapter, then regular_cloud bubble keeps per processing unit
(spiky-overlapping regions excluded -- conflict rule: validated spiky deletion
outranks a cloud keep), then spiky_cloud site deletions LAST via the
production-validated clean_spiky_region_clipped + background protected
interiors (the only in-panel delete authority). Reference architecture written
to docs/gen8_architecture.md.

Part 3 measured (chapters 002/004, all classifiers): 44 s/chapter; 10/9 spiky
sites (profile lists = production lists per the standing equivalence); 45/35
regular_cloud keep regions, 1/0 spiky-conflicts (expected 'thorn'-family
overlap, counted and resolved); spiky site deletions 133k/115k px; UPDATED
CHAPTER ADVERSARIAL: every in-panel deleted px lies inside a spiky site --
0 px outside on both chapters. Visual verification: spiky sites show production
semantics (thorn fringe deleted, protected text-bubble interiors kept); A1
region and damage-class regions identical to the panel-aware round. Commit
8.12.1.

## Diagnostic round: white-rectangle instances in composed chapter output -- source-material artifact, NO data loss, single shared mechanism (2026-08-11 20:53 EEST)

User review of 004's clean_chapter_full output found solid white rectangles
where content was expected. Diagnosis (diagnostic-only; NO code changes):

**Render purity proven first**: kept px in the red preview are byte-identical
to the source across BOTH chapters (0 mismatches) -- no compositing/blanking
bug exists and the 8.7.2-style stale-cache hypothesis is excluded for this
path. Source content is NOT lost anywhere; the white boxes are white IN THE
SOURCE.

**Verified instances** (kept white islands >= 5000 px with deleted
surroundings): 19 in chapter 004, 18 in chapter 002 -- pixel-identical shapes
at a constant ~165-row offset across the two chapters (e.g. 213x150/area
31950 in both; the 337x188 box at 004 y=77836 / 002 y=78011; the 689x88 strip).
These are RECURRING TEMPLATE elements of the release: the translation group
whited out the original-language caption boxes in the recurring recap/intro
segments and placed the Russian text nearby. The boxes are invisible in the
source (white on white, faint outline) and become conspicuous only when the
surrounding gutter is deleted.

**Keep attribution, uniform across all instances tested**: each box is kept by
a regular_cloud detection classified 'thorn' -- the faint irregular outline
reads as a thorn-family enclosed hole; the profile accepts it (thorn is not
rectangle-class, so the stroke-thickness gate auto-passes) and
clean_chapter_full's rc keep covers the box. Pocket-keep does NOT fire
(pocket_cover 0.00 -- the outlines are too faint to seal at G>=33). The boxes
were therefore DELETED in the panel-aware-only round (8.11.2) and became kept
when regular_cloud was wired in (8.12.1).

**Verdict**: single shared root cause; present in both chapters; zero data
loss (a display-conspicuousness effect over a pre-existing source blank).

**Proposal (NOT implemented, user's call)**: under the delete-bias priority,
filter regular_cloud keeps whose region interior contains essentially no ink
(e.g. < 1% px with G < 100): an empty hole protects nothing worth keeping;
real bubbles contain text and are unaffected. One variable, measurable on the
6 refs + both chapters. Until decided, the boxes render as kept white --
cosmetic, recoverable, safe direction.

Observed in passing, out of this diagnosis' scope: a few thin caption-glyph
fragments in gutters show partial tinting (sfx keep granularity on tiny
components) -- separate item, and UI-card handling remains the known deferred
gap. Commit: docs-only (this entry); no code changed.

## 8.12.4: empty-hole filter for regular_cloud keeps -- interior-ink metric, 57 empty keeps removed, real bubbles untouched (2026-08-11 21:07 EEST)

The a280337-proposed fix, applied at the COMPOSITION point in clean_chapter_full
(not in the profile: whether an empty detected hole deserves a keep is
orchestration policy under the delete bias; the profile's 8.5.1 behavior and
suite numbers stay untouched and its detections remain available).

Measured correction to the proposed "<1% region ink" spec: the flagship empty
box measures 3.3% ink REGION-level purely from its own 2 px outline -- region-
level density cannot separate. The shipped metric is INTERIOR ink: region inset
by max(6 px, 12% of the smaller dimension), then fraction of px with G < 100;
threshold RC_KEEP_INK_MIN = 0.01 (attempt 1).

Threshold verification (attempt-2 measurement, then STOP): the full interior-
ink distribution over all 80 rc regions in both chapters is bimodal at the 1%
bar -- empties at 0.0000-0.0004, everything at 1.40%+ verified VISUALLY to
contain real content (bubbles "ЭТО ГЁН...", "ДА...", the translated caption box
"1. БОЛЬ И...", art edges). Raising the threshold would strip real bubbles at
1.9-2.5% -- measured regress-elsewhere, NOT adopted; 1% stands.

RESULTS: 25 (004) + 32 (002) = 57 empty keeps filtered; rc kept px 691k -> 185k
(004) and 860k -> 225k (002); kept-white-island count 19 -> 4 (004), 18 -> 3
(002). The remaining islands are a DIFFERENT population than the complaint:
their keeps contain real translated text (the flagship box's rc region includes
the caption text at its bottom edge -- interior ink 1.40% is genuinely text;
others are dense-borderless segment keeps holding captions) -- correctly kept
under the architecture; visually they still read as partially-empty boxes
because the source is partially empty there.

Guards: chapter adversarial 0 in-panel deleted px outside spiky sites (both
chapters); 6-ref suite BIT-IDENTICAL with hard guard 0 px (this change cannot
touch that path; verified anyway); reference gutter bubbles clear the bar at
1.6-7.1% interior ink; full battery + 12-instance suite PASS identical.
Commit 8.12.4 (production defaults still untouched; clean_chapter_full remains
the opt-in orchestration).

## Diagnostic round (case A of A/B/C user-crop pass): caption text without backing box -- source has NO box; glyph-keep granularity erodes translated text (2026-08-12 08:53 EEST)

User crops (004_red chunks 29-36, window y63800-81400) show translated numbered
captions rendering on raw red with no/partial white backing. Measured on main
a4849a9, diagnostic only, no code changed.

Source truth first: the caption bands (cap1 y73490-73574, cap4 y77990-78045)
are 70-87% pure white in the SOURCE -- there is no drawn backing box under the
text; the whited-out template boxes sit ADJACENT (cap4's first-line left
portion overlaps the flagship y77840 box, rc-kept at interior ink 1.4%, which
reads as a "partial box"). NOT an 8.12.4 regression: the step-0 dump lists
every rc detection in the window (4 total); none spans the caption text, and
the one 8.12.4-removed empty box (y73145-73238 x18-104) is not under any text.
Text-on-red has existed since 8.11.2 wherever the band is gutter-typed.

The DEFECT is keep granularity, not the missing box: gutter treatment deletes
13.4% (cap1) / 16.7% (cap4) of the captions' ink px and 21.5% / 64.9% of their
midtone (anti-aliased stroke) px -- visible glyph erosion in the cleaned
output. Counter-instance cap5 (y79396) loses only 1.2%/0.0% because its
segment happens to be typed dense-borderless and is kept wholesale --
segmentation luck, not a caption-aware decision. Owner: sfx.py gutter
treatment / sfx_glyph keep granularity (composition); panel_segmentation only
determines which bands are exposed; background.py / frame.py uninvolved.
No PSD GT covers the window (viewports measured: 004 y2257-3225 / y3262-4141 /
y23415-24188 / y51457-52382 / y91980-92825) -- accounting is source-vs-output
with the established ink/midtone/blank conventions, flagged no-GT.
Recommendation: FIX family = caption text-line keep (protect glyph bbox rows
incl. anti-aliased skirt) -- same family as the 8.12.3 "tinting" side note.

## Diagnostic round (case B): torn white rectangles at spiky sites -- site action resurrects gutter blank inside its bbox; GT-quantified FN 24-27% at the 002 etalon sites (2026-08-12 08:53 EEST)

Symptom: kept white RECTANGLES with torn red interior contours at gutter spiky
bubbles (004 sites y71625-71964, y76251-76586; milder at y65914, y67453).
Detection is CORRECT (sites = production-equivalent lists; the spiky edge is
legitimate source style). The defect is composition-context: the production
site action `clean_spiky_region(_clipped)` wholesale-keeps sealed interiors +
protected px inside its bbox and CLEARS prior deletions there (validated
panel-context semantics, 12-instance suite). In GUTTER context this resurrects
blank the pass-1 gutter treatment had correctly deleted. Stage attribution
measured at site y71625: pass-1+rc had deleted 87,458 blank px inside the
bbox; the site action cleared 82,670 of them (added only 5,812 fringe px) ->
the visible white rectangle IS the bbox (blank kept 94.4% inside vs 36.7% in a
24px outside collar; site y76251: 88.0% vs 45.1%).

GT quantification (etalon PSD crops 002_5_y66008 / 002_6_y67551, full-width at
exactly the named offsets, established dual-denominator metrics, DARK_G=100):
FP-delete 0.092% / 0.117% white-only (fine, we exceed the manual clean almost
nowhere) but FN-delete 27.357% / 23.969% white-only (98,015 / 68,939 px): the
manual reference deletes EVERYTHING outside a smooth ellipse -- the expected
output is a clean soft-rounded balloon, not a spiky remnant in a kept box.

Family: NOT the Class-B occlusion-leak residual (leak under occluding art in
panel context); this is a new composition-seam family. Owner:
clean_chapter_full step 3 (sfx.py) applying the panel-context action in gutter
context; profiles/spiky_cloud.py and pipeline.py themselves unchanged-correct.
Recommendation: FIX in composition (site action must not clear pass-1 gutter
deletions outside the sealed bubble interior; optionally delete the full spike
zone in gutter context to match the manual reference). No code changed here.

## Diagnostic round (case C): silhouette contour-on-red + sliver-panel band -- content-bearing bands under gutter treatment; adversarial guard blind spot MEASURED (2026-08-12 08:53 EEST)

Symptom: thin double-line contour along a character's jaw/forearm (y65230-
65650) rendered on red. frame.py CLEARED: chapter_lines uses the morphological
inventory (detect_lines_morph, not the Hough path), and the only entries in
the window are real edges (h y65242 span 58-658; v x657 span y65241-65767) --
nothing traces the curved silhouette. Actual mechanism: the artwork band
y65368-65643 measures ink 0.029 vs DENSE_INK 0.15 (light skin / white shirt
art is MIDTONE, not ink) -> typed sparse-borderless -> gutter treatment:
73.7% of the window's midtone px and 20.4% of its ink px deleted; the kept
sfx_glyph outline strokes + their 2/4px Expand halo read as a double line.
Same family as case A (content in a gutter-treated band; keeps are stroke-
granular), distinct sub-cause (DENSE_INK misses midtone-dominant art).

Second instance, WORSE: band y78096-78700 typed panel with x-extent 642-663 --
_x_extent collapsed onto a 5-line v-cluster at x642-661 (the diagonal panel's
right border decoration; no left border exists), the known x-band-collapse
family recurring at SEGMENT-TYPING level (B1 fixed it at unit level only).
The band's content outside the sliver: 100,966 px deleted incl. 5,650 ink +
75,504 midtone (face/hair/background art visibly eaten in the render).

GUARD REVISION (honesty item): the chapter adversarial guard counts deletions
inside SEGMENTATION'S OWN panel/partial rects -- inside the 21px sliver it
correctly measures 0, while the 100,966 px next to it are structurally
invisible. The 8.11.2/8.12.1 "0 adversarial px" claims remain true as
specified but are WEAKER than previously presented: the guard cannot see
segmentation typing errors. Distinct from the Broken-ring class (ring broken
by art overlap; here the ring never existed on the left side).
Owner: panel_segmentation.py (_x_extent collapse; DENSE_INK typing) +
sfx.py keep granularity. No PSD GT in window (flagged). Recommendation: FIX
(guard should also bound content-px deletion per band; _x_extent needs
two-SIDED line evidence; DENSE_INK needs a midtone-aware art criterion).

## GEN 9 OPENED: port of the user's manually-perfected Photopea algorithm; two-classifier scope (2026-08-13 16:17 EEST)

Why a new generation: the gen7-vs-gen8 comparative investigation (testing
branch, b8c21f5) measured gen7's monolithic pipeline at FPink 0/0 on both
golds while gen8's classifier composition, even post-fixpass, held thousands
(gold001 27,576 / gold002 7,231). Gen8's failure mode was never one wrong
classifier -- it was independently-reasonable classifiers conflicting at
composition boundaries (authority ordering, extent collapse, default-delete
bias). Gen9 abandons that architecture.

Foundation: the user's new manual Photopea algorithm (.tmp/gen9/
new-pipeline.md), hand-tuned to parameter precision (Levels 33,1,34 --
38,1,39 and 51,1,52 tried and rejected; Threshold 226; Levels 248,1,249;
Threshold 178; Minimum/Maximum 1px). It is deterministic except exactly two
operator judgment calls: WHICH white regions are inter-frame background
(step 11 clicks), and WHICH glyph strokes at the frame/background boundary
get the select-expand(4px)-delete treatment (step 15 clicks). Gen9 = port
the deterministic part parameter-for-parameter + build exactly those two
classifiers (A: background selection, B: boundary-glyph selection), narrowly
scoped, no composition machinery. A third judgment call, if ever needed, is
a STOP-and-report, not a silent addition.

Plan-mode measurements against the user's own working PSD
(.tmp/gen9/002_1.psd: red / img+raster-mask / img-clone-1 / img-clone-2):
- GT self-consistent: img mask black == semi-etalon alpha transparent,
  100.0% agreement, 10,434,032 px (50.4% of the 690x30000 page). The
  workflow deletes the background FIELD (72.8% of blank) while touching
  only 1.3% of ink / 0.7% of midtone.
- Deterministic chain reproduced pixel-exact in read-only probes:
  clone-1 = Levels(33,1,34) per-channel -> Threshold(226) on Rec.709
  luminosity, 100.0% vs the PSD's own layer; clone-2 = Levels(248,1,249)
  -> Threshold(178) -> erode3x3 -> dilate3x3, 100.0%.
- POLARITY DISCREPANCY, file is authoritative: the written steps 20-30
  as literally read would delete panels; the final mask instead decodes as
  DELETE = dilate1_sq3(union of selected clone-2-white background comps)
  AND NOT clone1-black, plus 11 manual expand-4 SFX fills -- i.e. the
  fills in the notes are protective (mask built in inverted polarity).
  This formula reproduces GT at diff 17,194 px = 0.08% of the page before
  any classifier.
- Classifier A's GT here: of 1,165 clone-2-white components, exactly the
  14 FULL-WIDTH bands are deleted; 15 one-side edge slivers and all 1,141
  panel-interior whites kept; header block y<160 (margin + site banner)
  kept by operator convention. No sealed-pocket positives on this page --
  the user's ~50px thick-contour heuristic ships report-only, not tuned
  blind.
- Classifier B's GT here: 6 stroke groups >= 50 px (y2742-4504, y24261),
  glyphs crossing the frame border, deleted with ~4px expansion. User
  ruling (2026-08-13): the semi-etalon is UNDER-CLICKED -- B must recover
  these 6, and additional confident detections are reported as
  beyond-etalon candidates, not counted as FP.

Branch gen9 off main (8244233); code lives in src/gen9/ only; gen7/gen8
files and production defaults untouched; versioning 9.XX.YY; no merge
without user review. Success metric: end-to-end px-diff vs the PSD mask
as close to the 0.08% oracle ceiling as the classifiers allow, with the
established content-FP (ink G<100 / midtone 100-199) and FN metrics,
full-page only. Expected defect classes explicitly NOT chased this pass:
MinMax square-kernel steps on curves, spiky-cloud residue floating in the
cleaned field, dark/UI zones (flag and skip).

## GEN 9 DELIVERED: end-to-end port FN 0 / diff 0.087% page; A first-try 14/14, B decodes fills as sealed pockets 6/6 (+19 beyond-etalon) (2026-08-13 16:32 EEST)

9.01.00 (fdee008): deterministic chain pixel-exact vs the user's own PSD
intermediates -- clone-1 100.0% (Rec.709 lum), clone-2 100.0% (Rec.601
after Min/Max; the 3x3 pass washes out the weighting difference, the
empirically exact pair is pinned). Oracle ceiling (formula + the user's
exact selections): diff 13,087 px = 0.063% page, FP-ink 431, FN-ink 0.

9.02.00 (6bc3085): Classifier A = full-width-span rule + keep-top-band
(header/banner) convention. 14/14 GT selections, 0/15 edge slivers,
0/1,141 panel whites, first attempt. Sealed-pocket finder REPORT-ONLY
(no GT positives on 002_1; 2 candidates, 48/56 px).

9.03.00 (1451085): Classifier B -- measurement overturned the assumed
semantics: ALL six operator step-15 fills are SEALED BACKGROUND POCKETS
(0% stroke px, 77-95% clone-2-white; strokes stay via c1b protection),
not stroke deletions. Hole-topology discriminators failed (4/6 pockets
share panel holes -- the gen8 mega-component lesson recurring); the
user's thickness heuristic works at OBJECT level: compact (bbox<=250) +
thick (inscribed r>=3.0) ink comps in the 25px near-bg band = glyph
objects; their sealed pockets (area 30-3000, dist-to-bg 3-30px) join the
background selection in compose_delete. 6/6 GT + 19 beyond-etalon
candidates (user pre-ruled the semi-etalon under-clicked; all 19
visually verified as glyph text pockets, blue-boxed in the preview).

9.04.00: run.py end-to-end on 002_1: FN 0 (every manually-deleted px is
deleted), total diff 18,018 px = 0.087% page (3,853 = the 19 extras,
~14k = ring AA vs soft selection edge), FP-ink 755 / FP-mid 2,317.
Known defects reported not chased: MinMax square-kernel steps, floating
SFX residue 21,650 px / 1,801 fragments, dark/UI none seen. Constants
single-page-validated; 50px pocket proximity remains user estimate.
Confirmation: exactly two classifiers, no authority machinery. Report:
notes/reports/gen9_port_2026-08-13_report.md. Desktop previews delivered.
User gates: review of red preview (esp. the 19 blue-boxed extras).
