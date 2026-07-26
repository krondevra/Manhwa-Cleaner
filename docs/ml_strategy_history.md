# ML background-removal strategy history

Living document. Purpose: a fast way to check "has this been tried before,
and what happened" before repeating an experiment. Covers the deep-learning
era specifically (model 3.0+, Pepper & Carrot dataset); for the earlier
rule-based/classical-ML era and why the project moved off real-manhwa
training data entirely, see `docs/decisions.md` and `docs/history.md`.

Update this file whenever an experiment resolves (worked / didn't / why) —
that's the entire point of it existing.

## Core architecture (stable, don't relitigate)
- Small U-Net (PyTorch), binary segmentation, 7-channel input: RGB (3) +
  4 guidance channels from `ml_cleaner.py::make_guidance_channels()`.
- Guidance channels are a **local-contrast morphological gradient**
  (dilate−erode) + its closed/opened binary variants + Canny edges — not
  absolute brightness. Verified by direct code inspection (2026-07-07):
  mathematically polarity-invariant (`dilate(255−x) = 255−erode(x)`, so
  `gradient(255−x) = gradient(x)`; Canny operates on gradient magnitude,
  also inversion-symmetric). Any brightness/color shortcut the model learns
  lives in the RGB pathway, not here.
- Training data is 100% synthetic, from Pepper & Carrot (CC BY 4.0). **Real
  manhwa is never a training source — permanent policy, confirmed
  2026-07-06 after models 1.0-2.1 (trained on real chapters) were purged
  from git history for copyright.** Real chapters are inference-only.

  **The exact boundary of that policy, clarified 2026-07-22 (canonical
  statement — other docs defer here):**
  - **NEVER**: no real manhwa / copyrighted chapter pixels (including the
    manual-reference chapters under `.tmp/saved/chapters/` and everything
    under `data/chapters-initial/`) may contribute training signal to any
    model, directly or indirectly — no gradient updates, no
    pseudo-labeling, no distillation, no augmentation or synthetic data
    derived from them. This applies to fine-tuning any third-party model
    exactly as it does to training our own.
  - **EXPLICITLY OK** (established practice throughout this project's
    history — 085 regression crops, the 2026-07-10 manual-reference study,
    the 2026-07-22 GT quality numbers): using real chapters as **held-out
    evaluation targets** — running a trained checkpoint on them and
    measuring output quality (pixel comparison against the human-cleaned
    references, visual crop diffs, aggregate two-directional metrics).
    Same category as testing any tool on real-world input to measure it.
    No weights are ever updated from that measurement.
  - The evaluation-only files themselves stay out of version control
    (gitignored / `.tmp/`) as already practiced — the policy is about what
    is excluded from training, not about hiding evaluation material.

## The central recurring problem: isolating "background" from "content"
The same pixel color can be background or content depending on structure —
no rule-based or color-threshold approach generalizes. The fix that
actually worked, and everything since has built on or fought regressions
of:

### WORKED — explicit boundary marker (the v1.18.0 / v6 pivot)
Every isolated training variant (`framed_speechbubles_w` family,
`PepperNCarrotDataset/src/synthesize/synthesize_dataset.py::make_framed_variant`)
draws a fixed, explicit, contrasting 2px line at *every* content/background
boundary (`panel_edge()` — not just the page's outer edge, every panel-to-
gutter transition). The model learns "this line marks a boundary, delete
extends outward from it," not "background is color X."

**Why this was needed:** before it (v1-v5, pre-`v1.18.0`/pre-`3.24.1`),
training backgrounds touched all 4 image edges with no marker at all,
teaching the model that edge-touching background is normal — the
**flood-fill-leak** bug (real content connected to the image edge only
through other delete pixels gets eroded; `--reclaim-islands`,
`3.22.1`/`cf6a7e7`, is the inference-side postprocessing patch for the same
failure class, reclaiming delete regions not connected to an image edge).

### FAILED (twice) — flat/noisy binary context mask + gradient variants
**v7 attempt (`9393186`/`3.27.1`, `dd517e0`/`3.28.1`):** trained on
`framed_speechbubles_context` (flat white=keep/black=delete binary mask, no
texture, no boundary-marker line at all) + a gradient-background variant
whose delete region was the only place background was ever black. Model
learned a literal **"black ≈ delete" brightness shortcut** — deleted real
dark clothing/hair, kept real white margin it shouldn't have. Root-caused
by diffing v6.0 vs v7.0 red-preview crops on the same source pixels. Fixed
by excluding both from `BASE_VARIANTS` (still generated, just not trained
on — cheaper than deleting the generator).

**v9 attempt (2026-07-07 session):** tried fixing the *root cause* (not
just avoiding it) with `framed_speechbubles_context_textured` — same
keep/delete geometry, but both regions get per-pixel noise + a JPEG
re-encode, brightness polarity randomized per page, specifically to
decorrelate brightness from class. Trained a model on it (dataset-only
change, `--boundary-patch-ratio 0.0` ruling out a sampling confound) — got
the **same class of regression again**: real dark clothing/hair/background
deleted, confirmed on the same two real dark scenes used to catch the v7.0
failure. Root-caused (reasonably, not proven) to the textured context mask
specifically. Fixed the same way: excluded from `BASE_VARIANTS`.

**Working theory for why the context-mask family specifically keeps
failing, independent of the brightness-shortcut mechanism:**
`make_context_mask()`/`make_context_mask_textured()` are built directly
from the raw unframed render, **never routed through `make_framed_variant`**
— this whole variant family never got the v6 boundary-marker fix at all.
Every content/background transition in it is a raw, unmarked color
transition. Not yet tested in isolation (would need routing context masks
through framing, which reopens "what color line on a binary silhouette"),
deprioritized in favor of testing `framed_speechbubles_black` (direct
real-content-on-black, which *does* get the marker treatment) first.

### UNTESTED-BUT-NOT-IMPLICATED — solid-line black background
`framed_speechbubles_black` (2026-07-07): real content composited on solid
black, with a **near-white** (not pure white — `FRAME_ON_BLACK =
(235,235,235,255)`) 2px frame line, same mechanism as the working white-bg
design. Was present in the v9 training run that regressed, but
`--boundary-patch-ratio 0.0` and the context-mask exclusion isolate the
regression to the context mask specifically — this variant itself has never
been shown to cause harm, but also has never been trained in full isolation
(nothing else new in the same run) to confirm it's actually fine.

**Reference-image finding that reframes this (2026-07-07):** direct
inspection of real manga crops (`.tmp/black-background/` vs
`.tmp/white-background-white-frame/`) shows **real black-background panels
have no visible boundary marker at all** — flat black gutter blends
directly into dark content, unlike white panels which conventionally get an
outline + wide white gutter. The solid near-white frame line teaches a
convention that essentially doesn't occur in real dark content — plausible
independent contributor to poor generalization, on top of (or instead of)
any brightness-shortcut explanation.

**Update:** the sparse-tick variation of this same idea (real content on
black, marker present, just sparse instead of solid) was tested next and
also failed — see "FAILED (4th attempt)" below. That result weighs toward
the brightness-shortcut explanation being dominant over the missing-marker
one, which lowers expectations for this untested solid-line variant too,
though it's technically still the one combination never directly tested.

### FAILED (4th attempt) — sparse tick-marker boundary
`framed_speechbubles_black_ticked` — same idea as the solid frame line, but
a sparse dash/tick pattern (`ticked_panel_edge()` in `synthesize_dataset.py`,
`cv2.findContours(..., CHAIN_APPROX_NONE)` walked with an on/off period)
instead of a continuous line. Hypothesis: enough structural signal to avoid
the flood-fill-leak failure mode without teaching "look for a solid bright
line," forcing more reliance on the flat-vs-textured local-contrast
distinction the guidance channels already detect well. Implemented,
smoke-tested visually (clean even dashes, no corner artifacts), trained in
isolation (`--variants initial,framed_speechbubles_w,framed_speechbubles_w_jpeg,
framed_speechbubles_black_ticked`, `--boundary-patch-ratio 0.0`) and
evaluated on `data/chapters-initial/085.png` (2026-07-07).

**Result: regressed the same way as every prior black-bg attempt.** Real
dark content (background architecture, clothing, hair shadow) in a dark
dungeon scene was marked delete across large contiguous areas, matching the
v7.0/v9.0-context_textured failure signature closely. Per-variant val_loss
during training looked fine (`black_ticked=0.262` vs `w=0.240` at the saved
checkpoint — no red flag there), so this regression was **not visible in
the loss curve at all**, only in real-chapter visual inspection — reinforces
that loss numbers alone are not a sufficient regression check for this
class of failure (see methodology lessons below).

**`--reclaim-islands` on vs off made almost no visible difference** to the
regression — this is an important, new piece of evidence: it means the
failure is not primarily a flood-fill-leak/edge-connectivity problem
(which islands-reclaim specifically targets), but the model directly
predicting "delete" with high confidence across large contiguous dark
regions. This weighs the "missing/wrong boundary marker" hypothesis down
and the "brightness/darkness shortcut" hypothesis back up — even a
present, correctly-designed sparse marker didn't stop the model from using
raw darkness as a shortcut once black-background examples were in the
training mix at all.

**Also found: small contamination into white-bg dark-toned content.** One
of the three fixed white-bg regression crops (`white_bg_regression_crops.md`
crop C, a white burst SFX on white bg) showed a new small red intrusion
into a character's dark clothing/glove in the panel above the burst — a
region with no black background nearby at all. This is the first direct
evidence in this project's history of a black-bg training addition leaking
into white-bg-page quality via shared weights, not just failing on its own
terms. The two bubble-dialogue white-bg crops (A, B) showed no regression
and arguably slightly cleaner bubble edges than 8.0's output on the same
crops (unexplained, possibly noise, not adjudicated further).

**Updated working theory:** three of four attempts at black-background
training (flat context mask, textured context mask, ticked real-content
frame) have now failed with the same failure signature, each isolating
away a different specific mechanism (flat brightness correlation → noise →
missing boundary marker) without success. This is stronger evidence than
before that the issue may not be any single implementation detail, but a
more fundamental tendency for this architecture/training setup to learn
"large uniform-ish dark region → delete" once *any* real black-background
training data is present, regardless of how the boundary is marked — the
model may simply not have enough counter-examples (real content that's
large, dark-toned, AND correctly labeled "keep") to unlearn that
association within a dataset where black-bg is a ~9-10% minority share.
**`framed_speechbubles_black` (the original solid-line version, un-ticked)
remains the one variant never tested in a clean isolated run** — worth
noting it's the last easy variation left, but given the pattern across all
three tested mechanisms, expectations for it succeeding where the others
failed should be low without a different structural change (e.g. much
higher black-bg sampling weight, or a fundamentally different approach —
see open questions).

### FIXED (silent bug, not a regression) — JPEG-variant border painted before compression
`make_framed_jpeg_variant`/`make_transparent_framed_jpeg_variant`
(`synthesize_dataset.py`) painted the pure-black 2px border **before** the
JPEG round-trip (`JPEG_QUALITY=15`, aggressive). A thin black line against
white is exactly the high-frequency detail JPEG destroys at low quality —
the border in the actual saved training bytes was a blurred/ringing
approximation, not pure `(0,0,0)`, ever since this variant was added
(`v1.4.2`, `3.11.2`-era) — present through every white-bg-trained model
without being noticed. Fixed 2026-07-07 by stamping the border *after*
compression. Verified: border pixels are now exactly `(0,0,0)` with zero
variance. This variant has been in `BASE_VARIANTS` since v6, so this was
silently degrading the marker signal's reliability in every "working"
model version — didn't break white-bg (v6-v8 clearly still work), but is a
plausible minor contributor to boundary-precision issues generally (see
"clauds" below), and is exactly the kind of "is the marker actually pure in
the bytes the model sees" question worth checking for any future
marker-based variant.

## Other resolved issues (guidance channels / overlays)
- **Absolute-darkness guidance channel was blind to light ink on dark
  backgrounds** (`3.20.1`/`8116456`): the pre-fix channel only fired for
  `gray <= threshold`, so white ink on a black background scored zero
  contrast even though the boundary was fully visible in RGB. Fixed by
  switching to the polarity-symmetric local-contrast morphological gradient
  described above.
- **LANCZOS overlay resize smears hard alpha into a soft band**
  (`v1.6.5`/`cffd8cd`): SFX/bubble overlay resizing left a several-px-wide
  partial-alpha band baked into `_cleaned` targets, teaching an ambiguous
  boundary exactly where crispness matters most. Fixed by re-binarizing
  alpha to hard 0/255 after every resize (`_binarize_alpha()`).
- **Same-fill-color shapes had no visible edge** (`v1.17.0`): a same-color
  synthetic shape against matching real content had zero contrast for any
  RGB-derived detector to find — not fixable by a smarter guidance channel,
  needed a compositing-side fix (guaranteed contrast margin). Same root
  issue independently found for `sfx_blob` (flat near-black fill, no
  outline) and fixed the same way (every shape/overlay needs an outline in
  the *opposite* color from its own fill).
- **Thin frame borders eroding during inference** (`3.16.1`/`414157a`):
  `protect_frame_borders()` — inference-side heuristic protecting near-black
  pixels adjacent to kept content within the border band from erosion.

## Open / unresolved

### FAILED (6th attempt, informative) — model black-1.0, dedicated black-bg-only model
2026-07-09: after 5 attempts all training black-bg data as a ~9-10% minority share diluted
into a white-bg-majority dataset, and a reference-image finding that real black-bg manga
panels have no visible boundary marker at all, built a genuinely separate, dedicated model
(`data/models/black-1.0.pt` — new naming lineage, not part of the white-bg `N.0` series) on
**100% black-bg-composition data, zero white-bg dilution**, spanning a **marker-visibility
spectrum**: `framed_speechbubles_black`/`_black_ticked` (existing, near-white solid/ticked),
new `_black_gray`/`_black_gray_ticked` (mid-gray marker), new `_black_noline` (frame_color ==
bg_color, zero visible marker — matches real references most closely, still routed through
`panel_edge()`'s alpha-hardening, not the retired unmarked `make_black_variant()` helper).
`PepperNCarrotDataset` `v1.24.0`. `boundary_patch_ratio=0.0` (unchanged, isolating this from
the white-bg boundary-ratio finding). val_loss 0.319 → 0.110 (best, epoch 6); final
per-variant val_loss tight across all 5 marker levels (0.10–0.14), no marker-visibility
penalty visible in the loss numbers.

**Result: same failure signature as attempts 1–5 — and visibly worse.** On the two established
dark-scene crops (`085.png`, y=19226 villain/hood scene and y=21397 skeleton scene), red
(delete) coverage is 31.6% of the villain-scene crop, near-total across both real dark
figures — not just "large contiguous areas" like prior attempts, closer to "almost everything
dark gets deleted." `--reclaim-islands` on vs off changes coverage by ~0.6pp (31.6% → 30.9%),
confirming (as with the 9.0 sparse-tick attempt) this is a direct per-pixel brightness
misclassification, not a flood-fill/connectivity problem.

**Both leading hypotheses going into this experiment are now weakened, not confirmed:**
- **Shared-weight dilution** (the untested "reconsidering whether one shared-weight model is
  the right approach" idea from earlier writeups) — a fully dedicated model with zero
  white-bg dilution did not fix it. If dilution were the dominant factor, removing it
  entirely should have helped; it didn't.
- **Wrong/missing marker convention** — even `framed_speechbubles_black_noline` (zero visible
  marker, the variant that should most closely match real black-bg panels per the reference-
  image finding) failed the same way, and per-variant val_loss showed no meaningful
  difference across the marker spectrum.

**New, unproven hypothesis this result raises**: the regression looking *worse* with zero
white-bg exposure (vs. every prior diluted attempt) suggests any white-bg training data
present may have been providing some incidental regularization/anchoring against a
"large uniform-ish dark region → delete" shortcut, and removing it entirely let that shortcut
run unchecked rather than removing its cause. This points away from *dataset composition*
(dilution ratio, marker style) as the primary lever entirely, and toward something more
architectural — e.g. `pos_weight=4.0` (biasing the loss toward finding delete regions) combined
with the local-contrast guidance channels naturally producing low-signal gradients over large
areas of flat/shaded real dark art, not just over genuine flat background. Not tested; flagging
as the most promising next angle if this is picked up again, over further dataset-composition
variations.

**Recommendation: do not pursue further black-bg dataset-composition experiments (dilution
ratio, marker style) without new evidence — 6 attempts across both axes have now failed
identically or worse.** If revisited, test an architectural/loss-side change instead (e.g.
lower `pos_weight` specifically for a black-bg-only run, or a loss penalty that discourages
deleting large low-local-contrast-but-real-content regions), or reconsider whether black-bg
support is worth pursuing at all given 6 consecutive failures across every mechanism tried so
far. White-bg remains the sole recommended production domain (`10.0-baseline`).

### FAILED (confounded) — model 11.0-strips, manhwa-scroll dataset restructuring
2026-07-09: hypothesis was that Pepper & Carrot's per-page renders don't
match manhwa's single-column vertical-scroll convention (background only
top/bottom, never left/right) as well as assumed, and that stitching
conforming pages per-episode into long strips (closer to a real manhwa
chapter's continuous structure) would help. Built
`PepperNCarrotDataset/src/synthesize/page_conformance.py` (row-by-row
alpha x-run scan, flags pages with side-by-side/multi-column panel rows)
and `stitch_episode_strips.py` (drops non-conforming pages, concatenates
survivors per episode, slices into 3503px chunks). New, isolated sibling
tier (`renders_strips`/`dataset_strips`/`dataset_split_strips`) -- the
existing per-page tier and every checkpoint trained on it (3.0-10.0)
untouched.

**Real-scale finding that undercut the plan going in**: the full-dataset
classifier run found 67.1% of pages (188/280) non-conforming -- far higher
than the ~40% estimated from the original 10-page sample. 8/39 episodes
lost entirely (zero conforming pages). Only 92 pages survived to stitch,
yielding 98 chunks and a much smaller training set than the per-page tier
(528 vs 1506 train pairs, 60 vs 168 val pairs).

**Result: `11.0-strips` (same recipe as `10.0-baseline`, boundary_patch_ratio
held at 0.0, the only variable changed being the dataset tier) is clearly
*worse* than `10.0-baseline`** on every fixed crop tested -- all 3 clauds
crops and both standard white-bg crops show visibly larger red intrusions,
and the claw-mark stress crop is also worse. Not a subtle or ambiguous
result; consistent across every crop.

**This is a confounded experiment, not a clean test of the stitching
hypothesis** -- excluding 67% of pages cut training-pair volume by ~3x in
the same run that changed the stitching/chunking structure, so the
regression can't be attributed to "stitching doesn't help" specifically;
it's at least as likely simple data starvation. The original per-page
tier's own multi-column pages may also not have been as harmful as
hypothesized (the frame-border/panel_edge() isolation may already handle
left/right transitions adequately in practice, even if not by design).
**Do not adopt `11.0-strips`; `10.0-baseline` remains the recommended
production checkpoint.** If this idea is revisited, the two variables
(page exclusion, stitching granularity) need to be separated -- e.g. test
stitching on a size-matched subset of the per-page tier first, or salvage
non-conforming pages' conforming sub-regions instead of dropping them
whole, before concluding anything about the stitching mechanism itself.

### PARTIALLY WORKED — model 10.0, white-bg-only recipe simplification
2026-07-08: a 13-version, 6-crop comparison (v3.0-v9.0, all islands
variants) confirmed white-bg-with-border is the domain that actually
works, while black-bg has now failed 5 times (see below) and even v6.0/v9.0
regressed on dark content unintentionally. Decision: pause black-bg
entirely, focus model 10.0 purely on white-bg. `BASE_VARIANTS`/
`OVERLAY_VARIANTS` (`3.38.1`) dropped `framed_speechbubles_black`/
`_black_ticked`/`ui_black`, keeping only `initial`, `framed_speechbubles_w`,
`_w_jpeg`, `shapes_bw`, `shapes_mixed`, `ui_w`.

Two isolating runs: `10.0-baseline` (this recipe, `--boundary-patch-ratio
0.0`) and `10.0` (`--boundary-patch-ratio 0.5`, the first ever nonzero test
of that flag). Evaluated on a new fixed crop set
(`.tmp/notes/clauds_regression_crops.md`, 3 real "clauds" bubble instances)
plus the existing white-bg regression set.

**Result: `10.0-baseline` alone (recipe simplification, no sampling
change) is a real, visible improvement over 8.0** — smaller/gone clauds
intrusions on 2 of 3 crops, and cleaner rendering of an unrelated white
burst-SFX claw shape that both 8.0 and 10.0 partially bite into. **Adding
`--boundary-patch-ratio 0.5` (model 10.0) did NOT produce a further,
consistent improvement over 10.0-baseline** — on 2 of 3 clauds crops it
looks closer to 8.0's severity than to 10.0-baseline's improvement (a
bottom intrusion that 10.0-baseline eliminated reappears in 10.0). No
white-bg regression in either checkpoint.

Working theory for why boundary-patch-ratio didn't help as expected:
biasing that heavily toward boundary/curved-outline pixels may reduce the
diversity of clean, unambiguous "confidently white / confidently red"
training examples the model needs to build a strong local decision
function in the first place — over-sampling the hard cases without enough
easy-case grounding. Untested: a lower value (e.g. 0.2-0.3) might behave
differently; not yet tried. **Recommendation: keep `10.0-baseline` as the
production checkpoint, not `10.0`**, pending any follow-up boundary-ratio
tuning.

- **"Clauds" — imprecise, scalloped curved bubble-outline edges.** Present
  since v3.0. Confirmed via postprocessing tests (`--close-radius`/
  `--open-radius`) to be a genuine model-precision gap, not an
  inference-side-fixable artifact via those two flags. `--reclaim-islands`
  (a different, connectivity-based postprocessing flag) does substantially
  mitigate it in practice for most bubble instances — worth using by
  default for production output regardless of checkpoint — but it's a
  mitigation, not a fix: the underlying raw-model precision gap is still
  there, and doesn't catch every failure shape (see the `12.0` follow-up
  below for a case it doesn't fix). Three training-side levers tried and
  ruled out for the raw model: `--boundary-patch-ratio` at 0.5 (model 10.0)
  did not resolve it and looked mildly worse on 2/3 test crops; increasing
  model capacity `base_channels` 24→64 (model 12.0) measurably worsened it;
  and boundary-aware loss weighting (model 13.0, `--boundary-loss-weight
  5.0`) also worsened it, with a concrete identified mechanism (it compounds
  multiplicatively with `pos_weight` for delete-class boundary pixels,
  pushing the model toward *more* deletion exactly where precision matters
  most — see the `13.0` writeup for the untested fix this suggests). The
  white-bg-only recipe simplification (10.0-baseline) remains the only
  training-side change that helped — worth investigating why before trying
  another sampling-, capacity-, or loss-side fix (e.g. is it simply "fewer,
  more consistent variants → less competing signal", which would point
  toward dataset composition as the more promising lever over all three).
- **Black-background removal**, overall: unresolved after **6** attempts
  (flat context mask, noisy context mask, sparse-tick real-content marker,
  the accidental v6.0/v9.0 regressions, and now the dedicated
  100%-black-bg-composition `black-1.0` with a marker-visibility spectrum
  — see above, the most informative failure yet since it rules out both
  leading hypotheses rather than just adding another data point). Both
  "dilution" and "marker style" as the primary lever are now weakened by
  direct evidence, not just unconfirmed. Candidates worth considering if
  revisited: an architectural/loss-side change (lower `pos_weight` for a
  black-bg-only run, or a loss penalty specifically discouraging deletion
  of large low-local-contrast-but-real regions) rather than further
  dataset-composition variations, or reconsidering whether black-bg is
  worth pursuing at all given the pattern across 6 mechanisms. Black-bg
  training remains paused; white-bg (`10.0-baseline`) is the sole
  recommended production domain.
- **UI-box overlay generalization**: new variant family (2026-07-07,
  procedural sci-fi "system UI" HUD box), low dataset share (~11% per
  category), likely needs more exposure or more shape diversity before it
  generalizes as well as the long-established bubble/frame variants.

### FAILED (informative) — model 12.0, full-capacity (base_channels 24→64) U-Net on GPU
2026-07-10: with `10.0-baseline` established as production and both the recipe simplification
and `--boundary-patch-ratio` sampling change already tried for the "clauds" bubble-edge defect,
tested the one remaining untried lever: model capacity. `SmallUNet`'s `base_channels` went from
the project default 24 to 64 (the classic U-Net paper's channel progression, 64→128→256→512,
mid=768) — same architecture class, only the width changed. Also first-ever GPU training run on
this machine (`AMD Radeon 890M` iGPU via ROCm, `HSA_OVERRIDE_GFX_VERSION=11.0.0` required to work
around a MIOpen/BatchNorm JIT-compile failure on `gfx1151`). Recipe otherwise identical to
`10.0-baseline` (lr=2e-4, batch=2, patch=512, dice_weight=0.65, max_pos_weight=4.0,
positive_patch_ratio=0.70, `boundary_patch_ratio=0.0`, 10 epochs × 300 steps) — isolates capacity
as the single new variable. `--workers 4 --cache-size 4` (down from the project default 8/8),
since a first attempt at this capacity OOM-crashed at 8/8; 4/4 ran the full 10 epochs cleanly
with headroom to spare. Also did a dataset-hygiene pass alongside this run: deleted ~70G of
abandoned variant folders (`framed_speechbubles_gradient(_inv)`, `_context*`, `_black*`,
`_ui_black*`) already unreferenced by `BASE_VARIANTS`/`OVERLAY_VARIANTS` since `3.38.1` — no
effect on training composition, pure disk cleanup.

val_loss: 0.299 → 0.389 → 0.236 → 0.214 → 0.318 → 0.165 → 0.170 → 0.218 → 0.182 → **0.157 (best,
epoch 10)** — noisier trajectory than `10.0-baseline` but ended on its best epoch, not a
plateau/overfit pattern. Per-variant breakdown at epoch 10 tight (0.106–0.243), no outlier.

**Result: regressed on the clauds defect specifically, on the same 3-crop set used for
`10.0-baseline`.** Crop 1 (moderate top-notch instance): baseline shows a small top-center bite;
12.0 shows a much larger intrusion eating both the top *and* bottom of the bubble interior. Crop
3 (severe case): baseline's scattered small bites became a thick red ring wrapping nearly the
entire bubble outline in 12.0 — clearly worse, not better. Crop 2 was mixed/ambiguous (different
bite locations, not clearly better or worse than baseline). The 3 general white-bg crops (plain
white deletion, panel/gutter transition, white-SFX-on-white stress case) were visually unchanged
from `10.0-baseline` — the regression is isolated to curved bubble-outline precision, not general
white-bg handling.

**Capacity was not the bottleneck for the clauds defect, and increasing it measurably hurt the
clearest test cases.** This rules out "the small U-Net doesn't have enough capacity to draw a
precise curved edge" as the explanation — it now joins `--boundary-patch-ratio` (model 10.0) as
a tried-and-ruled-out lever for this specific defect. No tested hypothesis yet for *why* more
capacity hurt curved-edge precision specifically (untested guess: more expressive channels found
an easier, coarser edge-fitting shortcut with less pressure to nail the tight curve, given the
loss is not curvature-aware) — flagging as the open question if this is revisited, not something
to act on without evidence. **Recommendation: keep `10.0-baseline` as the production checkpoint.**
`data/models/12.0.pt` kept for reference, not recommended for use.

Real GPU timing data point (base=64, batch=2, patch=512, `--workers 4 --cache-size 4`): steady-
state ~1.7-2.7s/step, ~682-822s/epoch training-only, ~13-17min/epoch including blended val +
`--val-variants-breakdown`. One-time ~6.4min delete-ratio-estimation pass per run start (separate
from MIOpen warmup). Note: this run's wall-clock ("training finished in 392.9 min") includes an
unplanned ~3.5h laptop-suspend gap mid-epoch-3 — the process survived it cleanly (suspend
preserves process state, unlike a crash/reboot) and resumed automatically; real compute time was
~2.3h. Worth knowing if timing this again: `systemctl poweroff`/suspend during a background run
just pauses wall-clock, it doesn't lose progress, as long as it's sleep and not a power-off.

**Follow-up, same day: re-evaluated with `--reclaim-islands`, changes the picture.** The writeup
above never tested postprocessing — a broader multi-version, multi-spot comparison (via
`compare_models_video.py`, now with a `--screenshots` flag for saving individual comparison
frames) checked `10.0`/`10.0-islands`/`12.0`/`12.0-islands` side by side across many more spots
than the original 3-crop set. Two things came out of it:
- **`--reclaim-islands` closes most of the raw-model gap.** In most bubble-edge/UI-box spots,
  `12.0-islands` looks close to `10.0-islands` — the postprocessing fills in most of what the raw
  `12.0` model gets wrong. Without it, raw `12.0` was consistently worse than raw `10.0` across
  nearly every spot with a red intrusion, confirming the original finding — capacity increase is
  still not a fix for the underlying model's edge precision.
- **New finding, not caught by the original 3-crop set: a soft background texture (a diffuse
  smoke/dust effect) that `10.0-islands` reconstructs as one clean connected shape gets
  fragmented into many small disconnected specks in `12.0-islands`.** `--reclaim-islands` doesn't
  fix this one — it only reclaims delete-regions fully enclosed by kept content, and this is the
  opposite topology (many small keep-specks scattered through a delete region). So even with
  postprocessing, `12.0` isn't a clean win — it trades "worse raw bubble edges, mostly hidden by
  postprocessing" for "an occasional texture-fidelity regression postprocessing can't hide."

**Updated recommendation: `--reclaim-islands` should be the default for production cleaning
going forward, regardless of checkpoint** — it measurably helps `10.0-baseline` too, not just
`12.0`. Checkpoint choice is unchanged: keep `10.0-baseline`, don't invest further in capacity
increases without new evidence.

### FAILED (informative, mechanism identified) — model 13.0, boundary-aware loss weighting
2026-07-12: after dataset composition (helped), sampling ratio (`--boundary-patch-ratio`, no
improvement), and capacity (`base_channels` 24→64, regressed) were all tried for the "clauds"
defect, tested the one remaining training-side lever: the loss function itself. `DiceBCELoss`
weighed every pixel equally regardless of whether it's flat interior or a tight curved edge.
Added `--boundary-loss-weight`/`--boundary-loss-radius` (`src/ml_cleaner.py`): reuses the
existing `MORPH_GRADIENT` boundary detection (already used for `--boundary-patch-ratio`) to
build a per-pixel BCE weight map, dilated to a `--boundary-loss-radius`-px band. Verified via a
smoke test that `--boundary-loss-weight 1.0` is an exact no-op matching pre-change loss values.
Trained `13.0-boundaryloss` with `--boundary-loss-weight 5.0 --boundary-loss-radius 3`, otherwise
identical recipe to `10.0-baseline` (`base_channels=24`, same hyperparameters, 10 epochs × 300
steps) — isolates the loss weighting as the single new variable. val_loss 0.566 → 0.174 (best,
epoch 10), healthy-looking curve.

**Result: regressed, not fixed — and the raw-model gap was actually a bit worse than model 12.0's
in some spots.** Evaluated the same way as the model 12.0 follow-up (broad `compare_models_video.py
--screenshots` spot-check across 16 coordinates, not just the 3 clauds crops, both with and
without `--reclaim-islands`). Raw `13.0-boundaryloss` showed *larger* red intrusions than raw
`10.0-baseline` at nearly every bubble/UI-box spot checked — 2 of the original 3 clauds crops were
also worse, one was comparable. The 3 general white-bg crops were essentially unchanged (one
showed a small, isolated improvement on a single bubble's top curvature — not a consistent
pattern). With `--reclaim-islands`, both checkpoints again look similarly clean, same as model
12.0's pattern — postprocessing hides the raw regression rather than the fix actually working.

**Mechanism identified, not just "didn't work": the boundary weight and the existing
`pos_weight` (class-imbalance correction) compound multiplicatively for delete-class boundary
pixels in `F.binary_cross_entropy_with_logits`.** Effective weight on a boundary pixel is
`boundary_loss_weight` if the target is "keep" but `boundary_loss_weight × pos_weight` if the
target is "delete" (here `5 × 4 = 20x` vs `5x`) — a 4x asymmetry that wasn't accounted for in the
design, systematically pushing the model toward predicting *more* deletion specifically at the
pixels needing the most precision. This plausibly explains why intrusions grew instead of
shrinking, and is a concrete, actionable insight for anyone revisiting this: **a real next attempt
would need to either exclude boundary pixels from the `pos_weight` multiplier, or weight the
"keep" and "delete" boundary terms independently**, not just multiply a single scalar into the
existing weighted BCE. Not attempted here — flagging as the untested fix, not re-guessing the
same design blind.

**Recommendation: keep `10.0-baseline` as the production checkpoint.** `13.0-boundaryloss` kept
for reference, not recommended for use. This is now the third training-side lever tried and
ruled out for clauds specifically (sampling, capacity, and now boundary-loss-weighting as
naively implemented) — dataset composition remains the only thing that has helped.

### MIXED (informative) — model 14.0, SFX white-outline hypothesis + colored-SFX variant
2026-07-13: new hypothesis, untested until now: real manhwa SFX text typically has a white
outline that becomes invisible on white page background — structurally identical to a
speech-bubble's outline against its white interior. Theory: since the model shares weights
across the whole image, SFX-white-on-white gives an ambiguous/weak training signal that could
generalize into "thin light boundary near white background = uncertain, be conservative" —
leaking into bubble-edge (clauds) precision via shared weights. Not confirmed going in — one
hypothesis to test, single-variable isolation vs. `10.0-baseline`.

**Scope-changing finding before training even started**: no previously-trained variant actually
touched real-colored SFX at all. `framed_speechbubles_context_sfx` (the only generator output
compositing real SFX text) has been excluded from `BASE_VARIANTS`/`OVERLAY_VARIANTS` since the
v7.0/v9.0 brightness-shortcut regressions (built on the unsafe `make_context_mask()` foundation,
never routed through `panel_edge()`), and even where generated, `_apply_white_plan()` flattens
all SFX color to solid white. So testing this hypothesis required reviving a *properly-isolated*
colored-SFX variant into active training, not just fixing dormant code — confirmed with the user
before proceeding.

**Data-generation changes (PepperNCarrotDataset)**:
1. New `framed_speechbubles_sfx_w(+_cleaned)` variant, built on the `ui_w` precedent (safe
   overlay onto the already-`panel_edge()`-hardened `framed_speechbubles_w` base), not on
   `context_sfx`'s unsafe foundation. `make_sfx.py` (v1.25.0) now exports a per-(job,mode)
   outline-ring sidecar mask; `_paste_sfx_colored()` deletes the outline-in-target only where
   both the outline's own rendered color *and* the underlying page pixel are near-white
   (`luma >= 235` both ends) — narrow and structurally scoped, not a flat brightness rule. Input
   always keeps the full glyph; only the training target is simplified. Three-layer safety
   verification (permanent `validate_dataset.py` diff-against-baseline invariant, one-off
   connected-component leak audit, ep01 smoke test) all passed clean on the full 39-episode
   regen — no flood-fill-leak reopened.
2. `render_gradient()`'s outline color fixed to a hard per-pixel binary switch (black/white
   opposite the local fill luma) so gradient SFX never has a soft/ambiguous outline-vs-fill
   transition.

Trained `14.0` with `10.0-baseline`'s exact recipe (`base_channels=24`, same hyperparameters)
plus the new variant, nothing else changed. `--val-variants-breakdown` at epoch 10:
`sfx_w=0.26111` (clearly the hardest/highest of all variants, as expected for a new complex
one — next highest was `shapes_bw=0.22564`), `ui_w=0.15972` (mid-pack, not obviously elevated).
No prior run recorded a per-variant `ui_w` number to diff against directly, so the
shared-placement-math confound between `sfx_w` and `ui_w` (both go through
`_plan_overlays`/`_plan_sfx_overlays`) can't be fully ruled in or out from this run alone — but
`ui_w`'s absolute value gives no sign of it.

**Result: mixed, not a confirmation.** Evaluated on the 3 established clauds crops, 3 white-bg
crops, 2 new dedicated real-SFX-on-white crops (`.tmp/notes/sfx_regression_crops.md`, built
specifically for this test since no existing crop set covers colored/gradient SFX), and a broad
18-coordinate `compare_models_video.py --screenshots` spot-check, all against `10.0-baseline`
with and without `--reclaim-islands`.

- **Clauds crops (the actual target defect): no clear improvement on any of the 3.** Hypothesis
  not confirmed by the crop set built specifically to track it.
- **Dedicated SFX crops: no improvement either.** Crop B (red-to-black gradient SFX on white)
  unchanged. Crop A (blue swoosh with glow-to-white blur) got *worse* — raw `14.0` shows a new
  jagged/scalloped defect along a diagonal panel edge near the glow halo that isn't present in
  `10.0-baseline`.
- **White-bg crops: no difference on all 3**, including the closest prior analog (white
  burst-SFX claw shape).
- **Broad 18-coordinate spot check: 2 genuine improvements** (smaller bubble-corner intrusion at
  one coordinate than baseline; a red bite into a white shape below a black-stroke SFX mark that
  baseline had and `14.0` didn't), **1 mixed result** (better on one edge of a bubble, worse on
  another), **and 2 new regressions not present in `10.0-baseline`**: at two separate
  coordinates, raw `14.0` eats into legitimate white content (a snow/frost gradient background,
  and a speech-bubble-shaped white region) near SFX-adjacent or soft-textured content, where
  `10.0-baseline` stayed clean. Both new regressions are hidden by `--reclaim-islands`
  postprocessing (consistent with the established pattern for every checkpoint's raw-model
  gaps), so production-visible impact is smaller than the raw numbers suggest — but the same was
  true of model 12.0/13.0's regressions, which this project's methodology treats as real
  findings regardless. The remaining ~13 of 18 coordinates showed no meaningful difference. No
  diffuse-fragmentation regression (the specific failure mode that flagged model 12.0) was seen
  anywhere in this sweep.

**Interpretation**: the new regressions cluster specifically near soft/glow/gradient content
adjacent to SFX — the *opposite* direction from the original hypothesis (which predicted
under-deletion/conservatism near SFX, not over-deletion of legitimate soft content elsewhere).
Plausible mechanism: `sfx_w`'s much harder held-out loss (0.26111, well above every other
variant) may be pulling shared decoder weights toward more aggressive deletion decisions in
visually-similar soft/gradient contexts generally, not just at literal SFX-on-white pixels — a
different shared-weights side-effect than the one hypothesized, but still a shared-weights
effect. Item 3 from the original problem statement (SFX left with partial outline/residue,
overly conservative) is also not clearly resolved by this run — the two dedicated SFX crops show
no cleanup improvement.

**Recommendation: keep `10.0-baseline` as the production checkpoint.** `14.0` is not a clear
regression on the scale of model 12.0/13.0, but it is not the fix either — it trades a couple of
small, narrow wins for a couple of small, narrow new defects, with the core clauds problem
unchanged. Dataset composition (`10.0-baseline`'s original white-bg-only simplification) remains
the only lever that has produced an unambiguous improvement across four attempts (sampling,
capacity, boundary-loss weighting, now SFX exposure). Worth considering if revisited: the
shared-weights mechanism (in either direction) keeps showing up whenever a new, harder variant
is added — an architectural separation (e.g. a variant-conditioned decoder head, or excluding
distant regions from a single shared gradient update) might be a more direct way to test the
"shared weights leak conservatism/aggression across dissimilar regions" theory than adding more
training data ever can.

### FAILED (severe, informative) — model 15.0, auxiliary signed-distance-transform (SDT) head
2026-07-13: after four training-side/data-side levers on clauds (sampling ratio, capacity,
boundary-loss weighting, SFX exposure), all left `DiceBCELoss`'s fundamental nature unchanged —
independent per-pixel binary classification, no notion of "the predicted boundary should form a
smooth, geometrically consistent curve." This experiment targeted that directly: added an
optional second output head to `SmallUNet` (`out_sdt`, gated on construction so it's a true no-op
when disabled — zero new params/state_dict keys) predicting a per-pixel signed distance to the
nearest keep/delete boundary (clamped ±8px, normalized to [-1,1]), trained with an independent
additive smooth-L1 term (`--sdt-loss-weight`, deliberately *not* reusing `--boundary-loss-weight`'s
weight map, since that map's multiplicative interaction with `pos_weight` was the identified root
cause of `13.0`'s regression — kept the two mechanisms fully separate on purpose). Verified a true
no-op at `--sdt-loss-weight 0.0` (byte-identical `PatchDataset` outputs, no `out_sdt` keys in
`state_dict()`), smoke-tested at a low starting weight (0.2, per this project's own lesson from
`10.0`/`13.0`'s aggressive first attempts both backfiring) before the full run. Trained
`15.0-sdt` with `10.0-baseline`'s exact recipe otherwise, clean 109min run, no OOM, best checkpoint
at epoch 10 (val_loss=0.194, DiceBCE component only — SDT term deliberately excluded from the
checkpoint-selection metric to stay comparable to every prior checkpoint).

**Result: regressed, broadly and severely — the worst outcome of the four training-side levers
tried for clauds.** Evaluated the same way as every prior clauds attempt (3 clauds crops, 3
white-bg crops, plus the 18-coordinate broad `compare_models_video.py --screenshots` spot-check,
both with and without `--reclaim-islands`) and additionally against `--sdt-fusion` (the planned
inference-time mitigation, using the SDT head's own zero-crossing to refine the primary
classifier's boundary decision within a narrow band).

- **All 3 dedicated clauds crops got worse, unambiguously** — larger, more solid red intrusions
  than `10.0-baseline` at the exact same bubble instances in every case, not a subtle shift.
- **Broad spot-check: severe, wide-reaching regression, not narrowly confined to clauds.** Of 18
  coordinates: 1 showed a genuine improvement, ~6 showed no meaningful difference, and **10+
  showed new or substantially worsened red intrusions**, several severe — a bubble at one
  coordinate went from a modest scattered bite to nearly half its interior filled red; another
  showed deep erosion tearing into a previously-clean white character silhouette. Two coordinates
  reproduce specific failure modes seen in *other* checkpoints' regressions: one shows the same
  scale of catastrophic bubble-fill seen at `13.0-boundaryloss`'s worst spot, and — most notably —
  **the exact coordinate that caught `12.0`'s diffuse-fragmentation regression shows that same
  fragmentation pattern again here**, via a completely different mechanism (auxiliary geometric
  loss, not capacity).
- **`--sdt-fusion` does not mitigate the regression.** The fused output is visually indistinguishable
  from the unfused one. This makes sense in hindsight: unlike `--reclaim-islands` (pure
  connectivity logic, fully independent of anything the model learned), the SDT head shares the
  same trunk that produced the regressed primary head — it isn't an independent signal that could
  correct it, since both were shaped by the same training run.
- `--reclaim-islands` mitigates the visible damage as it always does for every checkpoint's raw
  gaps — production-visible impact is smaller than the raw numbers suggest, but this project's own
  methodology (established across `12.0`/`13.0`) treats the raw-model regression as the real
  finding, not something islands papers over into a non-finding.

**Mechanistic pattern worth naming explicitly: this is now the third distinct mechanism — after
capacity (`12.0`) and boundary-loss reweighting (`13.0`) — that adds extra emphasis on boundary
pixels during training and measurably WORSENS boundary precision, including reproducing the same
specific failure shapes (severe single-bubble fill, diffuse fragmentation) across otherwise
unrelated designs.** All three were careful, single-variable, well-motivated attempts targeting
different parts of the pipeline (network width, loss weighting, an auxiliary task) — none shared
an implementation bug with each other. The convergent failure suggests the common factor isn't in
any one mechanism's specific bug, but something more structural: this small U-Net's boundary
decisions may already be near a stability/capacity limit specific to *this* training data, and any
additional gradient pressure concentrated there — regardless of form — pushes it past that limit
rather than sharpening it.

**A concrete, previously-untested candidate explanation exists and points away from more
training-mechanism engineering: Phase 0 of this session's plan** (before implementing the SDT
head) found that `PepperNCarrotDataset`'s bubble shapes are not generated per-training-instance —
`make_bubbles.py` produces a small, fixed set of static assets once (`random.seed(42)`, run
once): 5 `oval_tail_*` variants (all the same `draw.ellipse(rw=195, rh=110)`, differing only in
tail angle), one `thought_bubble`, one `cloud_bubble`, one `burst_bubble`(+inv). At training time,
`synthesize_overlays.py` only ever applies uniform isotropic scaling (aspect ratio locked) and
repositioning — the outline geometry itself never varies. Across the entire 39-episode dataset,
the model has been exposed to roughly 8-10 distinct bubble curvature templates total, each
stamped down at different sizes thousands of times, while real manhwa bubbles vary continuously in
aspect ratio, size, and hand-drawn asymmetry. **This would explain the convergent pattern above
directly**: any training-side change that concentrates more gradient weight on boundary pixels is
concentrating it on a low-diversity, highly-repeated signal — closer to overfitting pressure than
to learning a general "trace an arbitrary smooth curve precisely" capability. This hypothesis was
flagged, not tested, this session (out of scope for the SDT experiment's own isolation) — it is
the most promising remaining lever for clauds, ahead of any further loss- or capacity-side
mechanism.

**Recommendation: keep `10.0-baseline` as the production checkpoint.** `15.0-sdt` is tracked in
git for reference only, not recommended for use — it is a clear regression, more severe and
broader than `12.0` or `13.0`. **Do not attempt a fourth training-mechanism lever (a different
loss shape, a different auxiliary task, a different capacity/depth change) without new evidence.**
Given three independent mechanisms have now converged on the same failure pattern, the next
experiment worth running is the bubble-shape-diversity fix implied by the Phase 0 finding above —
adding per-instance random aspect ratio and outline-jitter to `make_bubbles.py`'s generators
instead of reusing ~8-10 static templates — evaluated in its own isolated run, no loss/architecture
changes bundled in.

### FAILED (informative, mechanism identified) — model 16.0, training/inference resolution mismatch
2026-07-13, later same day: last session's plan for this slot assumed the clauds root cause was
bubble-shape template poverty in `make_bubbles.py` (~8-10 fixed static templates). **That premise
was checked directly this session and did not hold**: `framed_speechbubles_w` (the dominant active
training variant) is built from real, hand-drawn Pepper & Carrot bubble shapes extracted via
`process_speechbubbles.py` from the comic's own SVGs — `make_bubbles.py`'s templates aren't
consumed by any variant currently in `data/dataset/` at all. Bubble-outline stroke width (2px)
also already matches `panel_edge()`'s frame-marker width exactly, and training alpha channels are
already hard binary (0.000% intermediate-alpha pixels, 8-page sample) — the "soft transition" and
"marker inconsistency" hypotheses don't hold either.

**New finding, verified directly**: Pepper & Carrot training pages are uniformly 2481×3503px
(confirmed across 10 episodes). Every real chapter checked — evaluation chapters and the
human-cleaned manual-reference chapters demonstrating perfect achievable quality — is consistently
~690-720px wide. A ~3.6x scale mismatch, never addressed by any prior attempt. Since a CNN's
convolutional kernels operate at a fixed absolute-pixel receptive field, training at 3.6x the
linear scale of production input means every learned local curvature/boundary feature is
calibrated to a systematically gentler, lower-curvature-per-pixel signal than real chapters
present — a mechanistically distinct, well-motivated hypothesis, untested by any of the 5 prior
training-mechanism-side attempts.

**Implementation**: new `PepperNCarrotDataset/src/tools/scale_dataset_to_target_resolution.py`
resizes the actively-trained variants to a 690px target width (RGB via LANCZOS, alpha via LANCZOS
+ `_binarize_alpha()` re-hardening) into a new `data/dataset_scaled/` sibling tier — validated
clean (1953 pairs, 0 missing, 0 dimension mismatches, 0 soft-alpha pixels across a 130-page
spot-check). Also deleted the long-unused `framed_speechbubles_black*/_gradient*/_context*` families
from `data/dataset/` (excluded from training since `3.38.1`, zero effect on model behavior) — 106G
→ 37G, pure disk hygiene.

**First attempt — `16.0` with `--scale-jitter 0.2`**: added a training-time random per-patch
zoom augmentation to guard against the 690px target being estimated from only 4 real chapters (a
fair concern raised mid-session — not a robust sample of real-world webtoon export widths).
**Result: catastrophic regression, mechanism identified.** The zoom-out branch padded a shrunk
patch with a synthetic all-"keep" (white) border to fill the gap — teaching the model a real,
high-frequency lie (roughly half of all scale-jittered patches had an artificial "definitely keep"
border unrelated to real content) at high enough frequency to collapse the model's decision
boundary. Measured via a **two-directional pixel-ground-truth check** (islands-cleaned output as
ground truth for both true content AND true background, not just content as in prior sessions'
methodology): over-deletion of real content actually improved slightly (10.08% → 7.67% aggregate
across the 18-coordinate spot-check) — but **under-deletion of real background exploded from 0.00%
to 87.39%**, a failure mode the content-only metric used for models 12.0-15.0 would have completely
missed. This is now a permanent addition to this project's evaluation methodology: **always check
both directions** (content wrongly deleted AND background wrongly kept), not just one.

**Fix applied**: `scale_jitter_patch()` rewritten to only zoom in (crop, never pad) — zoom-out
draws are skipped rather than inventing pixels. Fixed in `src/ml_cleaner.py`, `--scale-jitter`
help text corrected to describe the new zoom-in-only behavior.

**Second attempt — clean retrain, `--scale-jitter 0.0` (isolating the core scale-match hypothesis
alone)**: bundling scale-jitter into the very first scale-match test violated this project's own
"one variable per run" discipline (methodology lesson #1) — corrected by retraining with the
augmentation fully off. Training was interrupted mid-epoch-8 by an external process kill (not OOM,
memory was never under pressure) partway through this session's time budget; epoch 7's checkpoint
(val_loss=0.088, the best of the run so far, healthy decreasing trend, tight per-variant spread)
was evaluated as-is rather than restarting, given the time budget. **Result: still a regression,
milder than the jitter-bug version but real and broad.** Two-directional measurement across the
18-coordinate spot-check: over-deletion 10.08% → 17.98% (worse), under-deletion 0.00% → 14.38%
(new problem, smaller than the buggy run's 87% but not zero). The 3 dedicated clauds crops showed
a similar mixed-to-worse pattern once measured in both directions (previously only over-deletion
had been checked, which showed an apparent improvement — see the methodology note above for why
that was an incomplete picture).

**A second bundled variable was identified, not yet tested**: `--patch-size` was left at 512
throughout, unchanged from `10.0-baseline`. On the original 2481px-wide pages, a 512px patch
covered ~21% of page width — a genuinely local crop. On the new ~690px pages, the same 512px patch
covers ~74% of page width — training patches stopped being local boundary crops and became
near-whole-page views. This plausibly changes what kind of training signal the model receives
(far less diversity of relative position/context per patch, `positive_patch_ratio`/
`boundary_patch_ratio` sampling logic designed around small local crops rather than whole-page
views) independent of whether the core scale-matching idea is sound. **Not tested this session**
(time budget) — if this hypothesis is revisited, `--patch-size` should be scaled down alongside
the dataset resize (e.g. proportionally, to preserve the original local-crop/page-width ratio)
as its own properly isolated follow-up, not bundled with the scale-match change again.

**Update — a third attempt (`17.0`) tested the patch-size confound directly and also failed,
in the opposite direction, closing out this experiment for the session.** `--patch-size` scaled
proportionally from 512 to 144 (matching the 690px page width, preserving the original
~21%-of-page-width local-crop ratio) — same scaled dataset, no scale-jitter, otherwise identical
recipe. Clean 10-epoch run, 6.3min (dramatically faster — patches are ~13x fewer pixels). Best at
epoch 5, val_loss=0.17716, healthy-looking curve, no crashes.

**Result: severe regression, opposite direction from `16.0`'s unscaled-patch attempt.**
Two-directional measurement across the same 18-coordinate spot-check: over-deletion exploded from
10.08% to **77.25%** aggregate (individual clauds crops as high as 69-73%), while under-deletion
stayed near zero (0.07%) — a clean, uniform collapse toward deleting almost everything, the polar
opposite failure mode from `16.0`'s "keep almost everything" collapse. Not a subtle or ambiguous
result — confirmed consistently across all 18 coordinates, not just the 3 dedicated crops.

**So both tested patch-size extremes fail, in opposite directions**: 512px (unscaled, patches
cover ~74% of the resized page) collapses toward under-deletion; 144px (proportionally scaled,
patches cover ~21%) collapses toward over-deletion just as severely. This rules out "scale
patch-size proportionally" as a simple fix, and is itself informative: patch-size interacts with
this resized dataset in a way that isn't captured by either naive choice tested. A plausible
uninvestigated factor: `--positive-patch-ratio`/`--boundary-patch-ratio`/`--min-positive-pixels`
were all left at their `10.0-baseline` defaults, un-scaled for the new patch geometry — at 144px,
`min_positive_pixels=256` is ~1.2% of the patch (a much higher bar relative to patch area than at
512px), which could itself distort what patches get accepted during sampling in ways that weren't
examined here.

**Recommendation: keep `10.0-baseline` as the production checkpoint.** All three `16.0`/`17.0`
attempts are documented; only the two non-buggy checkpoints are kept in git for reference
(`16.0.pt`/`.json`, `17.0.pt`/`.json` — the jitter-bug version was never kept). The core
scale-mismatch diagnosis (2481px training vs ~700px production, directly measured, not assumed)
is not disproven by this session's three attempts, but the "obvious" fixes (naive jitter, either
patch-size extreme) have now all failed with identified, non-overlapping mechanisms. **Do not
guess at a fourth patch-size value next time** — if this is revisited, it needs a properly
designed approach (e.g. sweep several patch sizes with adequate held-out evaluation before
committing to a single full run, or investigate the `min_positive_pixels`/sampling-ratio
interaction flagged above first) rather than another single-shot attempt at a middle value.

### NULL RESULT (inference-side, safe, kept) — `--repair-frames` enclosed-hole interior repair
2026-07-22: first new experiment in the inference-postprocessing family since `--reclaim-islands`/
`--protect-borders`. Idea (recalled by user from an earlier discussion, never previously written
down): the model draws frame/bubble strokes cleanly, and per the manual-reference finding
(`.tmp/notes/manual_reference_findings.md`) correct deletion is purely geometric — so any delete
pixel inside a region *fully enclosed by near-black strokes in the RGB* is wrong by definition and
can be forced back to keep. Targets the two interior failure topologies islands can't reach by
design: bites connected out to real background through other delete pixels, and keep-speck
fragmentation (model 12.0's defect).

Implemented as `repair_frame_interiors()` + `--repair-frames`/`--frame-darkness`/
`--frame-min-interior`/`--frame-inset` in `src/ml_cleaner.py`, running last in the postprocess
chain. Two designs were tested: a per-component "thin closed outline" detector (rejected — real
bubble outlines are 8-connected via tails/overlapping panel lines into filled dark art, so a
whole-component stroke-ratio test throws away good holes over guilt-by-association), replaced by
per-hole acceptance (any enclosed hole >= `--frame-min-interior` px; gutters/margins always touch
the strip edge so they flood-fill away and can never register as enclosed; provably one-directional,
delete->keep only). Synthetic smoke passed all cases: closed frame repaired, filled black rect
untouched, open frame skipped, merged bubble+blob component's hole still repaired.

**Result: exact no-op on every real chapter tested with `10.0-baseline` + islands.** Full-strip
mask diff on `085.png` (184k rows): 0 pixels changed. Two-directional pixel measurement against the
human-cleaned manual references `001`/`002` (first-ever eval against real GT, investigation-only
use): 0 pixels flipped on either. **Mechanism: for this checkpoint, the set of delete pixels inside
RGB-enclosed holes is empirically a subset of landlocked delete pixels — the model does not delete
a connected path across an intact dark stroke — so islands already reclaims everything this rule
can reach.** The hypothesized "bite crossing the outline stays edge-connected" topology does not
occur in `10.0-baseline`'s output. On the raw (pre-islands) mask the repair does fire (6,083 px on
one 4k-row slice), confirming the implementation works; islands simply gets there first.

Kept in the codebase anyway: zero cost and zero measured risk (proven byte-identical on production
output), provably can't add deletions, and it's an independent safety net if a future checkpoint
develops either target topology. Not part of the recommended production flags — those remain
`10.0-baseline` + `--reclaim-islands` alone.

**Follow-up, same day — raw mask (no islands), formal subset proof.** Ran the same GT evaluation
with `--repair-frames` applied directly to the raw model output. It fires substantially there:
467,899 px flipped on `001`, 189,133 px on `002`, cutting over-deletion 1.65%→1.17% and
1.29%→1.04% respectively (visual: an entire panel interior + bubble interior eaten by raw
intrusions comes back clean — `.tmp/repair_frames_eval/raw001_band_*.png`). Accuracy of its flips
vs GT: 100% right on `001`; 93.8% right on `002` (the 11,776 "wrong" flips are enclosed regions
the human deleted — islands wrongly reclaims those same pixels plus ~7x more, so frames is the
*more* GT-accurate of the two where they act). But **frames' flips are a measured 100% subset of
islands' flips on both chapters** — islands reclaims 2.3-3.2x more in total (it also catches
landlocked regions not enclosed by dark strokes in the RGB, e.g. bites into white content far from
any outline). So for `10.0-baseline`: `--repair-frames` is a strictly weaker partial substitute
for islands, not a complement — useful alone only if islands' topological assumption ever breaks
(e.g. an input where real background genuinely doesn't reach the image edge, or a checkpoint that
deletes across intact strokes), which is exactly the safety-net role it's kept for.

**Side finding worth more than the experiment — first direct GT quantification of production
quality** (vs. human-cleaned `001`/`002`, both directions, % of all pixels): over-deletion 0.61% /
0.56%, under-deletion **12.41% / 12.47%**. The under-deletion number is dominated by the known,
deliberate white-bg-only domain limit (the human removes black backgrounds; `10.0-baseline` by
design does not), but it puts a real number on that gap for the first time: roughly a quarter of
true background pixels in these chapters are background the current production domain doesn't
touch. If black-bg support is ever revisited, this is the baseline to beat, measured against real
ground truth rather than islands-cleaned pseudo-GT.

### PROBE (zero-shot, mixed — mechanism works, priors wrong) — CascadePSP pretrained refinement
2026-07-22: first experiment from the `.tmp/INSPIRATION/` papers review (item 1 in
`.tmp/notes/inspiration_papers_review.md`). Motivation: three independent trunk-side mechanisms
(capacity/12.0, boundary-loss/13.0, SDT/15.0) failed convergently on clauds — pointing at a
mechanism OUTSIDE the trunk. CascadePSP (Cheng et al. 2020, github.com/hkchengrex/CascadePSP) is a
class-agnostic refinement network (RGB + coarse mask → pixel-accurate mask), trained independently
on perturbed ground-truth masks, designed for train-low-res/apply-high-res mismatch. **Probe only:
pretrained weights, zero-shot, no finetuning, no changes to `10.0-baseline` or the dataset.**

Setup: isolated `.venv-cascadepsp` (CPU torch 2.13 + torchvision 0.28+cpu + pip
`segmentation-refinement`; the 2020 package runs fine on python 3.14 — one gotcha: torchvision must
come from the pytorch CPU index, the PyPI wheel mismatches and crashes on `torchvision::nms`).
Probe harness: `src/probe_cascadepsp.py` (spots / clauds / gt subcommands; foreground-mask
convention = keep = ~delete_mask, binarize refined soft mask at 127). Synthetic sanity: coarse
IoU 0.711 → refined 1.000. Runtime (CPU, full mode): ~5-37s per 900px window, 13-112s per
4000-row band, ~1.5h for both GT chapters.

**Results** (all vs `10.0-baseline`+islands production output):
- **3 clauds crops**: bubble outlines get visibly smoother/crisper — the refinement mechanism
  genuinely engages with line-art boundaries. Black-panel crop (clauds_3) untouched (0 flips).
- **18-coordinate spot-check (085)**: heavy one-sided churn — 620,991 px flipped keep→del vs
  89,243 del→keep. No GT on 085, adjudicated on the GT chapters below.
- **GT chapters (manual references, held-out eval per the clarified policy)**:
  | chapter | config | over-del | under-del | total err |
  |---|---|---|---|---|
  | 001 | islands | 0.61% | 12.40% | 13.01% |
  | 001 | +cascadepsp | 1.58% | 9.73% | **11.31%** |
  | 002 | islands | 0.57% | 12.45% | 13.02% |
  | 002 | +cascadepsp | 2.58% | 9.48% | **12.06%** |
  Flip accuracy vs the human etalon: keep→del 71.3% right (001) / 60.0% right (002);
  del→keep 79.0% right (001) / 26.9% right (002).

**The two faces, both verified visually:**
- **Win (looks like the etalon)**: gutter cleanup. Stranded white keep-specks around SFX glyphs
  and bubbles — the exact under-deletion topology neither `--reclaim-islands` nor
  `--repair-frames` can touch (both only un-delete) — get wiped to match the human etalon almost
  exactly, while the glyphs/bubbles themselves stay crisp
  (`.tmp/cascadepsp_probe/gt001_diff2_y65037_COMBO.png` — 4-way comparison incl. etalon). This
  also settled a convention question: the human DOES delete the white fluff around glyphs.
- **Failure (the whole over-del increase)**: full-bleed color panels. Low-texture interior
  regions (sky, cloud bands, sea) get carved out of real artwork
  (`gt001_diff0_y7850_*` previews) — natural-image saliency priors treating them as background.
  1.07M px (001) / 1.47M px (002) of destroyed content — and over-deletion is the worse error
  class for this pipeline (content loss is unrecoverable in output; under-deletion is leftover
  background a human can still erase).

**Honest adjudication**: net total pixel error DROPS on both chapters zero-shot — but the error
composition shifts toward content destruction, so this is **not usable as-is** in production. The
failure mode is semantic (wrong domain priors about what counts as an object), not mechanical —
where its object prior matches this domain (bubbles, SFX, gutters), output approaches the human
etalon; where it doesn't (full-color scenic panels), it hallucinates background inside artwork.
This is exactly the ToonOut-predicted natural-image→toon domain gap, and it is **consistent with
"needs P&C-specific training of the refinement scheme", not with "refinement is a poor fit for
this content"** — the probe's stated question. A P&C-trained refinement experiment (their
perturbed-ground-truth training scheme on our pairs) is now evidence-supported and worth scoping
as its own properly-designed experiment — NOT committed to by this probe, and out of scope here
per the probe's charter. Also noteworthy: zero-shot it already partially attacks both open
defect classes (keep-speck under-deletion; some of the background reduction sits in regions the
white-bg-only production domain deliberately never touches).

**Production recommendation unchanged: `10.0-baseline` + `--reclaim-islands`.** Probe artifacts:
`src/probe_cascadepsp.py` (tracked), previews/log in `.tmp/cascadepsp_probe/`.

### RESULT (positive, net improvement — not yet production-ready) — CascadePSP finetuned on P&C
2026-07-22/23: the follow-up experiment the zero-shot probe's adjudication called for. Scoped in
`.tmp/notes/cascadepsp_finetune_plan.md` (read that note for full design detail — training-pair
generation, stratified sampling rationale, hardware/compat findings); this entry records the
result. **Never trained on real manhwa** — P&C only, per the clarified policy above; manual-
reference chapters 001/002 used only as held-out evaluation, same as every other checkpoint.

**Setup**: cloned CascadePSP's own training code (`data/CascadePSP/`, third-party, not committed;
one compat patch needed — `models/sobel_op.py` hardcoded `.cuda()` on its Sobel kernels, removed).
Exported 1255 train / 140 val (image, clean keep-mask) pairs (`src/export_cascadepsp_pairs.py`)
from `data/dataset_split_scaled/` — only the variants `10.0-baseline` itself trains on
(`BASE_VARIANTS`/`OVERLAY_VARIANTS` minus `"initial"`), so the refiner sees the same domain as the
model it refines. CascadePSP's own `OnlineTransformDataset`/`boundary_modification.modify_boundary`
already synthesizes the coarse "seg" input online per sample (random IoU-targeted dilate/erode/
hole-punching) — no offline perturbation-pair generator was needed, simplifying the original scope.
Added `StratifiedRefinementDataset` (`src/train_cascadepsp_pc.py`) on top of that: crops are
centered on one of three strata (30/40/30 low_texture/boundary/uniform) instead of uniformly
random — `low_texture` targets flat/low-local-contrast KEEP interiors (the zero-shot probe's
carved-out sky/cloud/sea failure mode, must learn to RESTORE), `boundary` targets mask-contour
regions (the probe's genuine gutter/SFX win, must learn to preserve/extend it). Finetuned the
pretrained release weights (not from-scratch — dataset size matches the ToonOut finetuning
precedent, and the pretrained boundary machinery already worked in the probe; only the semantic
priors needed to shift).

**Two real bugs found and fixed during Phase 2 piloting** (both now permanent lessons):
1. `--workers > 0` silently defeated the whole stratified-sampling safeguard — each DataLoader
   worker forks its own copy of the dataset object, so `StratifiedRefinementDataset.counts`
   mutations never reached the main process (a 60-step pilot logged all-zero counts the entire
   run despite the design working correctly in isolation). Fixed by defaulting to `--workers 0`
   — justified since GPU compute (~10s/step) totally dominates 224px JPEG-decode cost, so
   multi-worker loading buys nothing here.
2. Redirected stdout is fully-buffered, not line-buffered — a 400-step pilot showed zero progress
   for 20+ minutes (looked exactly like a hang; diagnosed via `ps -o etimes=,time=` showing CPU
   time tracking elapsed time almost exactly, i.e. genuinely computing, not stuck). Fixed with
   `sys.stdout.reconfigure(line_buffering=True)` in both training and eval scripts — same risk
   class as this project's existing "silent OOM" monitoring lesson, different mechanism.

**Pilot (400 steps, corrected config): loss 6.66→~0.6-0.8, healthy convergence; real GPU timing
10.24s/step (890M iGPU, batch=2) — informed the Phase 3 step-count choice (4000 steps ≈ 11.4h
projected, an overnight-to-next-day run, not the "overnight" originally guessed pre-measurement).**

**Full finetune: 4000 steps, 14.1h wall-clock, loss converged to ~0.07-0.15 range (from 6.66),
strata mix held at 28.2%/40.3%/31.6% vs the 30/40/30 target throughout the entire run** —
confirms the stratified design worked as intended in the real training stream, not just in the
isolated dry-run. Checkpoint: `data/models/cascadepsp-pc-finetune-1.0.pth`.

**Evaluation (same three sets as the zero-shot probe, for direct comparison):**
- **3 clauds crops**: crops 1/2 show far less over-deletion than zero-shot (13,016/13,614
  keep→del vs zero-shot's 26,864/40,806) alongside more restorative del→keep activity; crop 3
  (black panel) still correctly untouched (0 flips), matching zero-shot and production.
- **18-coordinate spot set, aggregate**: del→keep 89,243→281,014 (3.15x more restorative
  cleanup), keep→del 620,991→286,960 (53.8% reduction in over-deletion) vs zero-shot. Not
  uniform across every coordinate — most spots improved substantially (several to near-zero
  over-deletion), a few improved only modestly, one (`y=120700`, already flagged in the
  zero-shot writeup as heavy two-directional churn) stayed essentially flat.
- **GT chapters 001/002, two-directional, the decisive numbers**:

  | | over-del | under-del | total error |
  |---|---|---|---|
  | ch001 islands (production) | 0.61% | 12.40% | 13.02% |
  | ch001 islands+cascadepsp zero-shot | 1.58% | 9.73% | 11.31% |
  | **ch001 islands+cascadepsp FINETUNED** | **0.37%** | **10.61%** | **10.98%** |
  | ch002 islands (production) | 0.57% | 12.45% | 13.02% |
  | ch002 islands+cascadepsp zero-shot | 2.58% | 9.48% | 12.06% |
  | **ch002 islands+cascadepsp FINETUNED** | **0.77%** | **10.85%** | **11.62%** |

  **Finetuned achieves the best net total pixel error of all three configurations on BOTH real
  GT chapters.** On ch001, over-deletion drops even below the production baseline (0.37% vs
  0.61%) — the finetune didn't just mitigate the zero-shot regression, it beat doing nothing.
  On ch002 it doesn't quite get there (0.77% vs production's 0.57%) but still cuts zero-shot's
  over-deletion by ~70%. Flip-accuracy vs GT: keep→del (additional cleanup) is precise, 91.1%/
  83.3% right on ch001/ch002 — most of the extra deletion is genuinely correct. del→keep
  (restoration) is markedly weaker, 48.1%/36.6% right — roughly half of what gets restored on
  ch001, worse on ch002, is wrong. **This asymmetry is exactly what explains the ch002 over-del
  regression vs production**: lower restoration precision there means more legitimate
  deletions get incorrectly undone.

  Visual confirmation of both directions (`.tmp/cascadepsp_probe_finetuned/COMBO_*.png`): a
  UI-box gutter case shows a completely clean win (stranded glow fragments fully wiped, boxes
  crisp, indistinguishable from a manual clean); an SFX-glyph case shows the same correct halo
  restoration alongside a spurious white column bleeding into real gutter background nearby —
  a concrete instance of the del→keep imprecision, plausibly CascadePSP's own blob-connecting
  behavior over-extending from a nearby high-confidence keep region.

**Honest verdict**: a real, measured net improvement over both prior configurations (production
alone, zero-shot refinement) on the only metric that matters (total pixel error against real
human ground truth) — the zero-shot probe's diagnosis ("needs P&C-specific training, not a poor
fit") is confirmed, not just plausible. **Not yet a production adoption**, for two concrete,
fixable reasons rather than a fundamental flaw: (1) del→keep precision (36-48%) is too low,
directly traceable to the still-imperfect restoration behavior visible in the crop above — a
lower `--strata-ratios` weight on `low_texture`, more steps, or an earlier checkpoint (a
mid-training checkpoint may generalize better than step 4000 if this is overfitting to the small
unique-page-content set) are the concrete next things to try; (2) CascadePSP inference itself is
heavy (multiple seconds to ~40s per production-sized window on CPU in this eval) — production
integration needs either GPU inference or a lighter-weight distilled model, neither built yet.
**Next step if continued: hold out a true validation split with tracked val loss (this run had
none — train loss only), sweep 2-3 checkpoints (e.g. step 1750, 2500, 4000) against the same
GT harness to find the best generalizing point rather than assuming later=better**, before any
production integration work.

Artifacts: `src/export_cascadepsp_pairs.py`, `src/train_cascadepsp_pc.py` (both tracked),
`data/models/cascadepsp-pc-finetune-1.0.pth` (gitignored, reproducible from the scripts + P&C
data), full logs/previews in `.tmp/cascadepsp_finetune_1.0.log` / `.tmp/cascadepsp_finetune_eval.log`
/ `.tmp/cascadepsp_probe_finetuned/`. **Production recommendation still unchanged: `10.0-baseline`
+ `--reclaim-islands`** — this result is a promising, well-evidenced research direction, not a
drop-in replacement yet.

### RESULT (small net win, partial — under-del target not met) — spatial ensemble of zero-shot + finetuned
2026-07-23: direct follow-up to the finetuned result above. The two checkpoints fail in
near-opposite ways — zero-shot cleans gutter/SFX halos well but carves real art (over-del
1.58%/2.58%); finetuned protects art well but leaves halos (under-del 10.61%/10.85%). Goal:
a deterministic combiner that takes finetuned's output as the safe base and admits zero-shot's
extra deletions only where they can't be real content, same philosophy as `--reclaim-islands`
(cheap, deterministic, composable). Full plan: `.tmp/notes/manual_clean_quality_plan.md`.

**Design** (`src/ensemble_refine.py`): label connected components of finetuned's KEEP mask; for
each component, compute the fraction of its pixels zero-shot marks DELETE. If that fraction
clears a threshold `F` and the component's area is under a cap `A` (small/medium blobs like
halos, not whole panels), zero-shot's deletions are applied inside that component. The area cap
is what protects large flat art regions even where zero-shot disagrees. A distance-gated
alternative (accept zero-shot-only deletions past some distance from agreed-keep territory) was
also implemented and screened — consistently 3-4x worse on over-del at every radius tried
(0/8/16px), confirmed by a synthetic adversarial test beforehand: when zero-shot disagrees with
100% of a region, no nearby agreed-keep anchor exists to protect it. Component gating was the
only viable rule going forward.

**First real bug, found by direct inspection (`gt001` y=65037)**: raw 8-connectivity fused a
33k-area SFX halo into the 474k-area panel directly above it (finetuned's own keep mask has no
gap between them) — a merged component this large fails any area cap outright, and the merge
also dilutes the zero-shot-disagreement fraction. **Fix**: erode the keep mask before labeling
(breaks weak thin bridges), then dilate a qualifying eroded component back — intersected with
the original keep mask — before the pixel swap, recovering close to the true pre-erosion extent.
Verified on this band: 0px changed (no fix) → ~26,000px changed (halo fully wiped, panel intact,
matching zero-shot's cleanup) after the erosion fix.

**Second bug, found the same way (`gt002` y=96627, discovered during the full 2-chapter eval,
not the 5-band screen)**: a steam/exertion effect near a character's head is fused with their
real T-shirt. At erode=0 this is one 73,718px component — correctly over the area cap, protected.
At erode=15 (the combo the screen had picked) it splits into fragments, and the fragment
containing the fused effect+shirt showed an *eroded* stats area of 44,507px — under the 60,000
cap — while its true dilated-back extent (the same computation already used for the pixel swap,
just not for the area check) was 62,120px, over the cap. **`ensemble_component()` was checking
the area cap against the shrunken eroded-label area, not the true extent** — an artifact of
erosion, not a real property of the region. This deleted real clothing, causing a measured
regression on chapter 002 (finetuned 11.6203% → ensemble 11.6768%, +0.0565pp worse). Read at
first as a fundamental "erosion radius is case-dependent" limitation; direct inspection instead
found a plain measurement bug. **Fix**: compute the dilated-back true extent for every component
(not just qualifying ones) and check the area cap against that, not the eroded stats area —
makes the cap erosion-radius-invariant. Verified directly on both decisive bands after the fix:
`gt001` y=65037 halo still swaps (~20,000px, mechanism intact); `gt002` y=96627 false positive
collapses to a 708px residual.

**Evaluation (GT chapters 001/002, two-directional, post-fix):**

| | over-del | under-del | total error |
|---|---|---|---|
| ch001 finetuned only | 0.37% | 10.61% | 10.98% |
| ch001 zero-shot only | 1.58% | 9.73% | 11.31% |
| **ch001 ensemble (component f=0.6 a=60000 erode=15)** | **0.37%** | **10.59%** | **10.97%** |
| ch002 finetuned only | 0.77% | 10.85% | 11.62% |
| ch002 zero-shot only | 2.58% | 9.48% | 12.06% |
| **ch002 ensemble (component f=0.6 a=60000 erode=15)** | **0.77%** | **10.84%** | **11.61%** |

Combined (pixel-weighted across both chapters): finetuned baseline total 11.2524% → ensemble
11.2382%, a real net improvement of -0.0142pp. Both chapters individually improve or hold; over-
del cost stays negligible (+0.0045pp combined) — no art-damage regression. This is a genuine,
if modest, win: strictly better than either checkpoint alone on the metric that matters, not
just a compromise between their failure modes.

**Honest verdict against the plan's own success criteria**: the over-del target (≤ finetuned's
0.37%/0.77%) is essentially met (0.37%/0.77%, within noise). The under-del target (≤ zero-shot's
+0.5pp, i.e. ≤~10.2%/10.0%) is **not met** — the ensemble closes only about half the gap between
finetuned and zero-shot's under-del (10.59%/10.84% vs the ~10.2%/10.0% target), because the area
cap that correctly protects large fused content (see the ch002 bug above) also means many
smaller-but-still-substantial halo/gutter regions that don't cleanly separate from art via
erosion never get admitted for cleanup. **Partial result, real and safe, not sufficient alone**
to reach manual-clean quality — the remaining halo gap is architectural to this rule (connectivity
+ area-cap can't distinguish "safe to clean" from "adjacent to protected content" any better than
the erosion radius allows), not a further parameter-tuning problem. Next candidate per the plan:
Phase B (checkpoint-sweep of the original finetune run, in progress) — does an intermediate
checkpoint dominate the tradeoff on its own, either as a production candidate or fed back into
this same ensemble as a stronger "finetuned" input.

Artifacts: `src/ensemble_refine.py` (tracked, commits 4.22.2 initial + 4.22.3 area-cap fix),
`.tmp/ensemble_refine/` previews, `.tmp/notes/manual_clean_quality_plan.md` (full execution log).
**Production recommendation unchanged: `10.0-baseline` + `--reclaim-islands`.**

### RESULT (informative, no actionable improvement — training is not reproducible here) — CascadePSP finetune checkpoint-sweep
2026-07-23/24: direct follow-up to the finetuned result's own open question ("sweep 2-3
checkpoints... to find the best generalizing point rather than assuming later=better"). The
original 4000-step finetune only kept its final checkpoint (a real bug — `train_cascadepsp_pc.py`
overwrote the same `--out` path at every save, discovered during review; **fixed in commit 4.22.1**
to also write `{out}.step{N}.pth` at every `--save-every`, permanent for all future runs of this
script). Since the original run's 15 intermediate checkpoints were unrecoverable, answering the
open question required a full 4000-step retrain — identical config/seed (seed 7, batch 2,
workers 0, lr 1e-5, ratios 0.3/0.4/0.3), `--save-every 500`, ~19.9h wall-clock (slower than the
original 14.1h due to CPU contention with Phase A's concurrent evals) — then evaluating all 8
resulting checkpoints (500/1000/.../4000) against both GT chapters (`src/probe_cascadepsp.py gt
--weights <ckpt>`), ~1-1.5h/checkpoint, ~10h total.

**Full two-directional GT results, combined across both chapters (pixel-weighted):**

| checkpoint | over-del | under-del | total error |
|---|---|---|---|
| step 500 | 0.64% | 12.15% | 12.80% |
| step 1000 | 0.60% | 11.15% | 11.76% |
| step 1500 | 1.84% | 10.40% | 12.25% |
| **step 2000** | **9.43%** | 9.75% | **19.18%** |
| step 2500 | 1.07% | 10.04% | 11.11% |
| step 3000 | 1.56% | 11.00% | 12.56% |
| step 3500 | 0.58% | 12.37% | 12.95% |
| **step 4000 (this sweep run)** | 0.68% | 10.28% | **10.96%** |
| step 4000 (original run, for comparison) | 0.54% | 10.71% | 11.25% |

**The trajectory is genuinely non-monotonic, not a smooth curve with a findable sweet spot.**
Step 2000 is a severe, reproducible outlier on both chapters independently (over-del 8.6-10.0%,
near-indiscriminate over-deletion) — checked and ruled out as a data artifact: the checkpoint
file size matches its neighbors exactly, loaded without warning, and the training loss logged at
that exact step (0.1851) was completely unremarkable, in the same range as every neighboring
step. **The synthetic online-perturbation training loss this project trains against does not
track this real-image failure mode** — a checkpoint can sit in a genuinely bad transient state
for real-domain behavior while its aggregate training loss looks fine, a new instance of this
project's recurring "the metric you're optimizing isn't the metric that matters" lesson.

**The critical, reframing finding**: this sweep's own step 4000 (10.96% combined) beats the
*original* run's step 4000 (11.25%) — two runs with identical seed and config produced
measurably different final checkpoints. Loss trajectories matched closely early on (e.g. step 500
loss agreed to 4 decimal places between runs), but by step 4000 real-image GT behavior had
diverged by more than most of the checkpoint-to-checkpoint differences found *within* either
single run. **GPU training here is not bit-reproducible even with a fixed seed** — plausible
sources are non-deterministic cuDNN/ROCm reduction kernels compounding over 4000 steps, not
verified further (out of scope for what this sweep was trying to answer). This means the earlier
within-run appearance of "step 2500 beats step 4000" (when step 2500 was compared only against
the *original* run's step 4000, before this run's own step 4000 had been evaluated) is better
explained as ordinary run-to-run training noise than as a genuine, exploitable mid-training
optimum — confirmed by the fact that **within this single run, step 4000 (the final checkpoint)
is in fact the best of all 8 points evaluated**, i.e. "later is better" held throughout this run;
there was no hidden earlier sweet spot to find once the full trajectory was mapped.

**Honest verdict**: the entire checkpoint-selection question is answered, but the answer carries
no actionable improvement. The ~0.1-0.3pp differences observed between candidate checkpoints —
across 8 checkpoints in a single run, and between two separate "step 4000" runs of the identical
recipe — are of the same order of magnitude as each other, meaning they're most parsimoniously
explained as training noise, not a discoverable tradeoff to exploit by picking the right step.
The only way to reliably do meaningfully better would be running multiple full trainings and
selecting the best by GT eval each time — a full ~14-20h + ~1.5h-eval cost per additional attempt,
poor return for gains this small. **No checkpoint change is recommended.** Total cost of this
investigation (original 14.1h finetune + this sweep's 19.9h retrain + ~10h of 8-checkpoint GT
eval, ~44h combined) for a verified-but-marginal (~0.1-0.3pp) result and one permanent, genuinely
useful tooling fix (checkpoint saves no longer silently overwrite each other) — flagged plainly
as a poor cost/signal ratio in hindsight, not spun as a win. **No further checkpoint-selection or
ensemble-parameter tuning is planned without new evidence** — a materially better result, if
pursued, would need to address the training data itself (see Phase C in
`.tmp/notes/manual_clean_quality_plan.md` — not started, requires explicit go-ahead given its own
much larger cost and unverified preconditions).

Artifacts: `src/train_cascadepsp_pc.py` (4.22.1 fix, tracked), `.tmp/run_checkpoint_sweep_eval.sh`,
`data/models/cascadepsp-pc-finetune-1.0-sweep.step{500,1000,...,4000}.pth` (gitignored, ~271MB
each, reproducible from the script + P&C data + seed 7), `.tmp/sweep_step{N}/` previews,
`.tmp/notes/manual_clean_quality_plan.md` (full execution log). **Production recommendation
unchanged: `10.0-baseline` + `--reclaim-islands`.**

### RESULT (SFX-exposure training pilot + GPU speed + first production integration point)
2026-07-25: three threads from one overnight session, summarized here; full detail split across
`.claude/plans/snazzy-cuddling-creek.md`, `.tmp/notes/cascadepsp_sfx_exposure_plan.md`, and
`.tmp/notes/cascadepsp_production_integration_plan.md`.

**GPU inference** (`--device cuda` added to `src/probe_cascadepsp.py`/`src/ensemble_refine.py`,
commit 4.22.7): a real, substantial speedup nobody had tried — full-precision mode ~40-55min/chapter
(~2-3x vs the CPU baseline's 60-120min), `fast=True` mode ~2-3min/chapter (~17x) for a small
(~0.17pp) quality cost. **CPU vs GPU inference confirmed bit-identical** (0px difference across
the base model, zero-shot, and finetuned refiner) — device choice changes speed only here, not
segmentation decisions, unlike training (see methodology lesson #9). Also found and documented
(methodology lesson #10): CascadePSP's output depends on how much surrounding context a crop
carries, not just its weights — a small window and the full `GT_BAND`-sized band `run_gt()`
actually uses are not interchangeable.

**SFX-exposure training pilot**: the CascadePSP finetune had never seen a single composited SFX
pixel during training (crisp or blurred) — a real, previously-unnoticed gap. Two bounded
continuation runs (400 steps each, from the checkpoint-sweep's step 4000) tested whether closing
that gap helps. Full two-chapter GT results, combined total error: sweep baseline 11.2508% →
step400 11.1333% → step600 10.9666% (best) → step800 12.9245% (sharp reversal, stopped there per
an explicit pre-agreed condition). **Step 600's aggregate win is not a clean one** — direct visual
inspection (caught by the user on a side-by-side comparison image, confirmed by pixel diff) found
it introduces a real, new, small-content-deletion defect (a motion/SFX mark near a character's
sleeve, correctly kept by both the baseline and step400, incorrectly deleted by step600) that the
aggregate metric doesn't penalize enough to show up. **Adopted step400 for production**, not the
higher-scoring step600, specifically because of this — the smallest, most conservative change
footprint from the baseline.

**First production integration** (`--cascadepsp-refine` added to `ml_cleaner.py process`/
`process-folder`, commit 4.22.9): opt-in flag, default off, applies CascadePSP last after all
existing postprocessing (matches how every quality number here was ever measured), default
checkpoint `cascadepsp-sfx-pilot.step400.pth`, default `fast=True`. A real bug was found and fixed
during smoke testing (not just assumed correct): the first implementation ran CascadePSP on an
entire chapter in one pass; every quality number in this project's history used banded processing
instead, and the single-pass version broke exactly as methodology lesson #10 predicts — it
restored a large blank gutter to "keep" that `--reclaim-islands` alone correctly deleted. Fixed by
reproducing `run_gt()`'s exact banding inside the new integration code. Full GT validation of the
integrated command (not just the standalone refiner in isolation) on both held-out chapters:
within 0.01-0.04pp of the already-measured standalone numbers. End-to-end wall-clock through the
actual `process` command: 3min1s / 2min17s per chapter.

**Honest bottom line**: `--cascadepsp-refine` is now a real, working, validated opt-in option —
not a research probe anymore. It is NOT a claim that it beats `10.0-baseline` + `--reclaim-islands`
alone in every case; it's a slower alternative with a modest, real quality profile and its own
known tradeoffs (small new-defect risk at the margin, as found above). **Production default
unchanged: `10.0-baseline` + `--reclaim-islands`.** `--cascadepsp-refine` is available for anyone
who wants to try the alternative.

### FAILED (severe, informative, 4th confirmation of a known mechanism) — model 18.0, self-contained coarse+refine head (RefineHead)

2026-07-25: after deciding against any third-party pretrained weights — including CascadePSP
itself, closing that entire thread (`.tmp/notes/cascadepsp_production_integration_plan.md` stays
as an opt-in, non-default option; not pursued further) — this experiment tried to reproduce
CascadePSP's coarse-then-refine idea *inside* `SmallUNet`, entirely from scratch, trained only on
Pepper & Carrot. Plan: `.claude/plans/snazzy-cuddling-creek.md`.

Added `RefineHead` (gated like `out_sdt`, zero-cost when disabled) taking the coarse decoder's own
full-resolution features (`u1`), its raw coarse logits, and a compressed/upsampled bottleneck
summary (global context) — a deliberate application of methodology lesson #10 (a refinement stage's
output depends on how much context it sees). `SmallUNet.forward()` split into `coarse_forward()` +
the existing heads so training could substitute a synthetic "coarse mistake" mask
(`synthesize_coarse_mask_perturbation()` — own from-scratch erode/dilate/hole-punch implementation,
not CascadePSP's `modify_boundary`) in place of the real coarse prediction. Stage A: warm-started
from `10.0-baseline.pt`, coarse decoder frozen (confirmed via `load_state_dict(strict=False)` — the
only missing keys were `refine_head.*`, 0.98% of total params trainable). Stage B: bounded 400-step
pilot (`src/train_refine_head.py`), `boundary_patch_ratio=0.5` (deliberately biased toward
boundaries — the whole point of the head), loss dropped steadily 0.72→0.29 with the usual bounce,
no plateau — a clean mechanical result on its own.

**Qualitative check on the established hard-case crops (`.tmp/notes/white_bg_regression_crops.md`,
`sfx_regression_crops.md`, `src/probe_refine_head.py`, both heads run through the identical tiled
`predict_delete_mask` + `--reclaim-islands` pipeline) told a different story: severe regression.**
Across 5 windows, refined output flipped 70,430px keep→delete against only 33px the other way. Two
of the three bubble/curved-outline crops show new, unambiguous scalloped red intrusions eating real
bubble interior at the top curvature — reproducing the exact "clauds" defect shape `10.0-baseline`
had already fixed at these instances (confirmed: the coarse-only output, which is numerically
`10.0-baseline` unchanged, is clean at the same crops; only the refined output is damaged). SFX/flat
crops were largely unaffected.

**This is not a new failure mode — it's the 4th independent mechanism to hit the exact same
convergent pattern already named in model 15.0's writeup**: capacity (`12.0`), boundary-loss
reweighting (`13.0`), an auxiliary geometric task (`15.0`'s SDT head), and now a dedicated
structural refine stage trained with boundary-biased sampling — every mechanism that concentrates
extra gradient pressure or attention on boundary pixels measurably worsens this small network's
boundary precision instead of improving it, regardless of how differently each mechanism is built.
`RefineHead`'s architecture was specifically designed to sidestep the previous three mechanisms'
own explanations (not a loss reweighting, not raw capacity, has real access to global context) and
still reproduced the same failure — meaningful additional evidence for 15.0's structural hypothesis
over a mechanism-specific one.

**Recommendation: do not continue this design.** No Stage C (joint finetune) or self-distillation
variant is likely to fix this — both would still concentrate training signal on boundary
correction, the same property all four failed mechanisms share. `18.0-refinehead` is not tracked in
`data/models/` (pilot checkpoint only, `.tmp/`, not committed) — nothing to revert in production.
**Production unchanged: `10.0-baseline` + `--reclaim-islands`.**

**Correction (2026-07-25, caught before acting on it): the paragraph above originally suggested
`make_bubbles.py`'s ~8-10 static bubble-outline templates as "the most plausible remaining path" —
this was stale language recycled from the pre-16.0 framing and is wrong. Model 16.0's own entry
above already found `make_bubbles.py`'s templates are dead code, never consumed by any variant in
the active training pipeline. A follow-up corpus check this session confirmed the real bubble-shape
source (`process_speechbubbles.py`, hand-drawn shapes from Pepper & Carrot's own translator SVGs)
isn't diversity-starved either: ~2,475 genuinely distinct outlines across the 39 episodes, no
template/symbol reuse. Both diversity-count hypotheses are now ruled out. The next thing actually
being checked (see `.claude/plans/snazzy-cuddling-creek.md`) is a distribution-shift question
instead — not how many distinct shapes, but whether real manhwa bubbles present *tighter curvature*
than anything in the P&C training range — a different, still-untested hypothesis.**

### FAILED (informative, third bubble-shape hypothesis ruled out) — real/training curvature distribution check

2026-07-25, same day: followed up on the distribution-shift question the correction above raised —
not diversity count, but whether real manhwa bubbles present tighter curvature than the P&C training
range covers. `src/probe_bubble_curvature.py`: rendered the isolated bubble/SFX-only SVG layer for 5
bubble-dense P&C episodes (61 pages), measured discrete radius of curvature per oval-classified
shape; found real bubble interiors the same way `repair_frame_interiors` finds enclosed frame/bubble
interiors (flood-fill a dark-stroke component's padded bbox from its corner). Curvature measured in
raw pixels at each side's native resolution (P&C ~2481px pages, real chapters ~690-720px),
deliberately not scale-normalized — a CNN's kernels have a fixed absolute-pixel receptive field
(model 16.0's own finding), so this is the comparison that matters.

**Result: not supported.** P&C oval-bubble min-radius-of-curvature distribution (106 shapes):
`[5,10,25,50,75,90]` percentiles = 7.4, 7.6, 10.9, 13.7, 16.6, 19.1px. The 3 already-documented,
confirmed clauds defect instances (`.tmp/notes/clauds_regression_crops.md`, chapter `085.png`),
ranked against that distribution: 11.4px (28th percentile), 14.2px (57th), 200.8px (100th — looser
than every sampled P&C training bubble). **None of the 3 confirmed real-world failures are curvature
outliers relative to training.** Verified by eye against the crop notes' own bubble-text descriptions
before trusting the numbers, not just accepted at face value — a real methodology snag was caught and
fixed in the process: the whole-page real-chapter aggregate also picks up rounded-corner panel frames
as "enclosed holes" (no clean size/shape cutoff separates them from real bubbles — confirmed areas
overlap directly), so that aggregate was excluded from the conclusion; the per-instance clauds
measurement doesn't share this problem (small, single-bubble-focused crops) and was checked visually
before use. Full writeup: `.tmp/notes/bubble_curvature_check.md`.

**This is the third bubble-shape-related hypothesis ruled out this session** — template diversity
(dead code), corpus diversity (not starved), and now curvature range (not exceeded). Combined with
5 prior training-mechanism attempts on this defect (models 12.0, 13.0, 15.0, 18.0-refinehead, plus
16.0/17.0's resolution-mismatch attempts), the shape/geometry angle looks exhausted for now — no
further bubble-shape-related lever is currently identified as worth trying without new evidence.

**Direction check-in, same day** (`.tmp/notes/full_auto_direction_2026-07-25.md`): with this angle
exhausted, asked directly rather than guessing from earlier, partly-superseded statements. Result:
still targeting full automation (not settling for an assisted workflow), the third-party-weights
policy is **narrowed, not reopened wholesale** — open to a cleanly, fully auditable MIT/CC-licensed
option specifically (ToonOut/`MatteoKartoon/BiRefNet`, MIT code+weights, CC BY 4.0 training data),
not CascadePSP itself (its own upstream data licensing was never fully clarified) — bounded pilots
only, and quality is now the explicit tiebreaker over speed/safety-conservatism (a shift from the
original four concerns, where frame-damage risk was weighted alongside quality). Next proposed step:
a cheap zero-shot ToonOut probe, not yet started.

### PROBE (zero-shot, failed — severe, both directions) — ToonOut/BiRefNet

2026-07-25, same day: the proposed next step above, executed. `src/probe_toonout.py` (mirrors
`src/probe_cascadepsp.py`'s structure exactly — same `SPOT_YS`/`SPOT_H`/`MARGIN`/clauds-crop
constants; `GT_BAND` reduced to 1200 for this model specifically, see below). Weights:
`joelseytre/toonout` (MIT, finetuned on a CC-BY-4.0 1,228-image anime dataset) on
`ZhengPeng7/BiRefNet`'s architecture (MIT code), loaded via `transformers
.AutoModelForImageSegmentation` + a stripped-prefix `load_state_dict`. New isolated venv
`.venv-toonout` (ROCm torch, same build as `.venv-cascadepsp`; hit and reused the same
`torch.backends.cudnn.enabled = False` MIOpen workaround found earlier tonight). Zero-shot only —
no P&C exposure, no finetuning, matching the bounded-pilot time budget for this session.

**Methodology note**: ToonOut/BiRefNet resizes every input to a fixed 1024x1024 square regardless
of aspect ratio (its own official demo does this) — an extreme distortion on a
`GT_BAND=4000`-row x 690px-wide band (~6.7:1). A one-band sanity check at that scale was run
first and showed a severe failure (see below); `GT_BAND` was reduced to 1200 (~2.6:1, closer to
the aspect ratios `SPOT_H`/clauds-crop windows already use) for the real GT eval, to give the
model a fairer test than the worst-case aspect ratio alone.

**Result: severe, unambiguous failure, worse than production on both error axes at once.**

| chapter | config | over-del | under-del | total err |
|---|---|---|---|---|
| 001 | islands | 0.79% | 12.37% | 13.16% |
| 001 | +toonout | 9.33% | 26.13% | **35.47%** |
| 002 | islands | 0.58% | 12.48% | 13.06% |
| 002 | +toonout | 10.84% | 25.59% | **36.43%** |

Roughly **2.7x worse total error than doing nothing**, and unlike CascadePSP's zero-shot result
(which traded over-deletion for under-deletion in a legible, single-direction way — content
destruction in exchange for gutter cleanup), ToonOut got worse on **both** axes simultaneously —
not a rebalancing, just wrong. Confirmed visually before trusting the numbers (this session's own
repeated lesson): one window had its entire dark-background panel deleted (real art, correctly
kept by islands); another had its entire white gutter kept untouched; a third had an entire,
clearly-legible speech bubble deleted outright. The 18-coordinate spot-check aggregate matches:
1.24M px flipped delete→keep and 2.47M px flipped keep→delete across just 18 windows — over 30%
of sampled pixels disagreeing with the islands baseline in one direction or the other, with no
consistent bias.

**Diagnosis**: not a mechanical bug (weights loaded cleanly, 0 missing/unexpected keys; inference
ran without error) — a genuine domain mismatch. ToonOut/BiRefNet is trained for single-subject
anime *character* cutouts (one salient figure against a background), not multi-panel comic pages
with bubbles, gutters, and mixed full-color/line-art content. It has no coherent notion of
"background vs. content" for this page structure at all — confirmed by the inconsistent failure
direction (sometimes keeps everything, sometimes deletes everything, sometimes deletes only the
one clearly-salient object and nothing else) rather than a single, correctable bias.

**Recommendation: do not pursue a P&C finetune of ToonOut.** Unlike CascadePSP's zero-shot
failure (a legible, single-direction bias — natural-image saliency priors over-trusting scenic
low-texture regions — that finetuning plausibly could and did correct), this failure has no
consistent direction to correct via finetuning. **Production unchanged: `10.0-baseline` +
`--reclaim-islands`.** Full artifacts: `src/probe_toonout.py` (tracked), previews/log in
`.tmp/toonout_probe/`, `.tmp/logs/toonout_gt.log`.

## Generation 6 pivot (2026-07-26): third-party weights closed out, full self-synthesis begins

With RefineHead (self-contained, model 18.0), three bubble-shape/curvature hypotheses, and both
third-party refinement options exhausted, the user decided on a full pivot: no more P&C-composition
tuning or third-party weights of any kind — a 100%-self-authored synthetic-data generator and
staged curriculum training run instead (~2-3 day budget). Full plan:
`.tmp/notes/synthetic_curriculum_plan.md`. Both third-party options investigated this session are
closed out here as **deliberate, evidence-based rejections, not abandoned dead ends**:

- **CascadePSP: rejected for licensing-provenance reasons, not quality.** Its P&C-finetuned
  checkpoint (`cascadepsp-pc-finetune-1.0-sweep.pth`) posted the best net total pixel error of any
  configuration ever tested on both real GT chapters (10.98%/11.62%, beating both production
  13.02%/13.02% and its own zero-shot result 11.31%/12.06%) — a real, measured, positive result.
  It was set aside only because its base ResNet50 weights' own upstream training-data licenses
  (DUTS/ECSSD/FSS/MSRA_10K and similar mixed-license photo sets) were never independently audited
  by the CascadePSP authors, which this project's now-narrowed-but-still-real third-party-weights
  policy does not accept regardless of the wrapper repo's own license. The 16 checkpoint files
  (~4.1GB) remain on disk, gitignored, kept rather than deleted, in case of future reference.
- **ToonOut: rejected for both a genuine quality failure and the same licensing concern.**
  Zero-shot GT eval was ~2.7x worse total error than production, failing in both over-deletion and
  under-deletion simultaneously with no correctable direction (see the PROBE entry immediately
  above) — a real quality rejection independent of licensing. Its base BiRefNet weights carry the
  same class of unaudited-upstream-lineage concern as CascadePSP's ResNet50 base.

Both efforts' code, probes, training scripts, and checkpoints-that-are-git-trackable remain fully
reachable on the local `archive` branch (created at commit `130ad9f`, the last commit before this
pivot) — nothing was deleted from git history, only some already-gitignored scratch directories
were cleaned off disk (`data/CascadePSP`, `data/dataset_split_scaled`, `data/refinement_pairs`,
`data/refinement_pairs_sfx`, `.venv-cascadepsp`, `.venv-toonout`, ~41.6GB).

**Generation boundary**: the CascadePSP/ToonOut refinement era (originally generation 5,
`5.1.1`-`5.7.12`) is folded into generation 4 on `main` (`4.16.1`-`4.22.12`, content-identical
renumber; the original 5.x-numbered commits are preserved unchanged on the `archive` branch,
tip `130ad9f`, which is why generation 5 does not otherwise appear on `main`) and closes at
commit `4.22.12`. Generation 6 (fully self-synthesized curriculum, no P&C composition tuning, no
third-party weights) begins at `6.1.1`. See `docs/decisions.md` for the versioning-scheme entry.

## Methodology lessons (apply these before starting a new experiment)
1. **One variable group per training run.** Every regression that was hard
   to attribute (v7, v9) involved bundling multiple simultaneous dataset
   changes. When adding N new things, isolate at least the ones with any
   plausible interaction risk into separate runs.
2. **Any new background-color variant needs an explicit boundary marker,
   proven present in the actual saved bytes** — not just intended by the
   code. The JPEG bug above shows intent and actual bytes can silently
   diverge; spot-check pixel values after generation, not just after
   writing the function.
3. **Visually smoke-test new synthetic generators on one page before a full
   39-episode regen.** Caught a real corner-notch geometry bug in
   `make_ui_boxes.py` this way, and confirmed the tick-marker pattern looked
   right before committing to a full regen.
4. **Standard regression check: diff real red-preview crops against the
   same source pixels, across versions.** This is how every regression in
   this history was actually caught and root-caused — not by loss numbers
   alone (loss magnitude isn't even comparable across dataset-composition
   changes).
5. **Real manhwa reference images: never training pixels; inspection and
   held-out evaluation are both fine** — see the canonical policy boundary
   in the "Core architecture" section at the top of this doc. Two genuinely
   useful findings came from *looking* at real references (the sci-fi
   UI-box style, the "real black panels have no border line" insight), and
   the 2026-07-22 GT quality numbers came from *measuring* against the
   human-cleaned references — neither ever feeds pixels into the dataset
   or gradients into a model.
6. **Per-variant val_loss is not sufficient to catch this class of
   regression.** The tick-marker attempt's `black_ticked` val_loss looked
   completely unremarkable (in line with other variants) on the exact
   checkpoint that then failed badly on a real chapter. Always do the
   real-chapter visual check before trusting a checkpoint, regardless of
   how healthy the loss curve looks.
7. **`--reclaim-islands` on/off is a useful diagnostic, not just a
   postprocessing option.** If a regression looks the same with islands
   reclaim on or off, it's a direct per-pixel misclassification (points at
   a brightness/color shortcut); if islands reclaim measurably fixes it,
   it's a connectivity/flood-fill problem instead. Run both whenever
   diagnosing a new dark-content regression.
8. **Pixel-ground-truth measurement must check BOTH directions: content
   wrongly deleted AND background wrongly kept.** Model `16.0`'s
   scale-jitter bug produced a checkpoint that looked like a genuine
   improvement on a content-only metric (over-deletion actually dropped)
   while catastrophically under-deleting real background (87% wrongly kept
   white, completely invisible to that one-directional check). Every
   pixel-ground-truth evaluation from here on should measure both
   `red & gt_white` (content lost) and `white & gt_red` (background kept)
   against the same islands-cleaned ground truth, not just one.
9. **A fixed seed does not guarantee a reproducible GPU training run.**
   The CascadePSP checkpoint-sweep retrain (see above) used an identical
   seed/config to the original 4000-step finetune and matched its loss
   trajectory closely early on, but the two runs' final checkpoints
   diverged enough in real-image GT behavior (~0.3pp total error) to
   change which one looked "better" — a difference of the same order of
   magnitude as the checkpoint-to-checkpoint variation being investigated
   within a single run. Before attributing a small (<0.5pp) difference
   between two training runs to a real cause (a hyperparameter, a data
   change, a "better" checkpoint), check whether it's within this
   run-to-run noise floor first — it may not be signal at all.
10. **CascadePSP's output depends on how much surrounding context the crop being
    evaluated carries — a tightly-cropped window and the full `GT_BAND`-sized band
    `run_gt()` actually uses are NOT interchangeable, even with identical weights and
    device.** Found 2026-07-25 while investigating why a fresh comparison image didn't
    match an older one on the same crop/checkpoint: CPU vs GPU inference was verified
    bit-identical (0px difference across the base model, zero-shot, and finetuned
    refiner — device is not the risk here), but a small `MARGIN`-padded window
    (~3000 rows) around just the region of interest gave CascadePSP a different global
    downsampled context than the full ~4600-row `GT_BAND`-sized band `run_gt()` computes
    inference on before cropping the interesting region out — producing genuinely
    different segmentation decisions (2.8% of pixels differed on one test crop, not
    subtle noise). One direct consequence discovered the same night: an entire earlier
    finding ("SFX-exposure training pilot fixes two real defects", based on tightly-
    cropped before/after comparisons) had to be retracted once redone with correct
    context — the true effect was ~27x smaller than what the undersized-window crops
    showed, because most of the apparent "fix" was actually cross-training-run noise
    (lesson #9) leaking in through a second, compounding methodology error. **Always
    reproduce `run_gt()`'s exact band-and-margin construction (or run the real thing)
    when building a comparison crop — never approximate it with a tighter window "close
    enough" to the target region.**
