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
