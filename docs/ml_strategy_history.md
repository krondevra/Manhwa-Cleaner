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
(`notes/clauds_regression_crops.md`, 3 real "clauds" bubble instances)
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
crops, 2 new dedicated real-SFX-on-white crops (`notes/sfx_regression_crops.md`, built
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
(`notes/manual_reference_findings.md`) correct deletion is purely geometric — so any delete
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
`notes/inspiration_papers_review.md`). Motivation: three independent trunk-side mechanisms
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
`notes/cascadepsp_finetune_plan.md` (read that note for full design detail — training-pair
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
(cheap, deterministic, composable). Full plan: `notes/manual_clean_quality_plan.md`.

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
`.tmp/ensemble_refine/` previews, `notes/manual_clean_quality_plan.md` (full execution log).
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
`notes/manual_clean_quality_plan.md` — not started, requires explicit go-ahead given its own
much larger cost and unverified preconditions).

Artifacts: `src/train_cascadepsp_pc.py` (4.22.1 fix, tracked), `.tmp/run_checkpoint_sweep_eval.sh`,
`data/models/cascadepsp-pc-finetune-1.0-sweep.step{500,1000,...,4000}.pth` (gitignored, ~271MB
each, reproducible from the script + P&C data + seed 7), `.tmp/sweep_step{N}/` previews,
`notes/manual_clean_quality_plan.md` (full execution log). **Production recommendation
unchanged: `10.0-baseline` + `--reclaim-islands`.**

### RESULT (SFX-exposure training pilot + GPU speed + first production integration point)
2026-07-25: three threads from one overnight session, summarized here; full detail split across
`.claude/plans/snazzy-cuddling-creek.md`, `notes/cascadepsp_sfx_exposure_plan.md`, and
`notes/cascadepsp_production_integration_plan.md`.

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
itself, closing that entire thread (`notes/cascadepsp_production_integration_plan.md` stays
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

**Qualitative check on the established hard-case crops (`notes/white_bg_regression_crops.md`,
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
confirmed clauds defect instances (`notes/clauds_regression_crops.md`, chapter `085.png`),
ranked against that distribution: 11.4px (28th percentile), 14.2px (57th), 200.8px (100th — looser
than every sampled P&C training bubble). **None of the 3 confirmed real-world failures are curvature
outliers relative to training.** Verified by eye against the crop notes' own bubble-text descriptions
before trusting the numbers, not just accepted at face value — a real methodology snag was caught and
fixed in the process: the whole-page real-chapter aggregate also picks up rounded-corner panel frames
as "enclosed holes" (no clean size/shape cutoff separates them from real bubbles — confirmed areas
overlap directly), so that aggregate was excluded from the conclusion; the per-instance clauds
measurement doesn't share this problem (small, single-bubble-focused crops) and was checked visually
before use. Full writeup: `notes/bubble_curvature_check.md`.

**This is the third bubble-shape-related hypothesis ruled out this session** — template diversity
(dead code), corpus diversity (not starved), and now curvature range (not exceeded). Combined with
5 prior training-mechanism attempts on this defect (models 12.0, 13.0, 15.0, 18.0-refinehead, plus
16.0/17.0's resolution-mismatch attempts), the shape/geometry angle looks exhausted for now — no
further bubble-shape-related lever is currently identified as worth trying without new evidence.

**Direction check-in, same day** (`notes/full_auto_direction_2026-07-25.md`): with this angle
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
`notes/synthetic_curriculum_plan.md`. Both third-party options investigated this session are
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

### Stage 1 (frames) accepted limitation: SFX-on-white-background over-deletion (2026-07-29)

Stage 1 (frames-only diagnostic pipeline, closing checkpoint `.tmp/checkpoints/stage1/a6_full10k/a6_full10k.pt`)
is marked done **with one documented open issue**, not fully clean, per the plan's decision
gate (one focused diagnose+fix attempt max before accepting and moving on).

**The issue**: a real SFX glyph rendered with an outline over a blank white background
(`ch1_sfx_text` ROI, `.tmp/diagnostics/ch002_rois.json`) gets 52.4% mean delete-probability
on `a6_full10k.pt` — well past the `KEEP_PROB_CEILING=0.30` threshold `regression_suite.py`
uses to gate "keep"-expected content. Originally user-flagged 2026-07-28 ("SFX with outline
on white bg which we delete caused flood fill leakage"), diagnosed the same day: the
full-width top/bottom-border frame convention itself is correct (pre-existing, not a new
bug), production's postprocessing (`--reclaim-islands`) can't fix it (its edge-connectivity
heuristic can't distinguish this from real background touching the page edge), and the
actual production checkpoint (`10.0-baseline.pt`) doesn't show this behavior at all.
Confirmed a training-scale-artifact hypothesis directionally (2k→10k measurably reduces the
effect — visually, the flooding "pulled back" from face/neck/chin content between the 2k and
10k checkpoints) without fully eliminating it by 10k images.

**Why accepted rather than re-chased today (2026-07-29)**: this ROI's exact numeric gate
result was only discovered today (2026-07-28's "12/12 checks passed" closing-gate run did not
actually include this ROI — see `synthetic_curriculum_plan.md`'s 2026-07-29 Part A section
for the gate-coverage gap this exposed, now fixed going forward). The underlying defect
itself was already diagnosed yesterday with a full investigation (real production-pipeline
test, real production-checkpoint comparison, scale-ablation via 2k vs 10k) — that counts as
today's "one focused attempt" already spent, so it is not being re-diagnosed from scratch.
**Next real attempt should try**: more training data (the clearest working lever so far,
consistent with the training-scale-artifact hypothesis) via the 30k-scale full run already
planned for Part 6/7, and/or revisit whether this ROI structurally resembles the
`ch1_caption_box_in_splash` finding (also isolated bordered/outlined content on a busy
background) closed by Stage 2 bubble training — if so, an SFX-specific Stage may close both
at once (see `notes/stage3_sfx_hypotheses.md`'s hollow-shape hypothesis, which already
flags this exact connection and its skin_neck-shortcut risk).

**`ch1_caption_box_in_splash`: CONFIRMED CLOSED (2026-07-31), resolved by Stage 2 bubble
training, already in production.** Re-verified directly against `regression_suite.py --cases
ch1_caption_box_in_splash` on the actual production checkpoint (`.tmp/checkpoints/stage2/
b2_bubbles_2k_prestage/b2_full2k_finetune.pt`): PASS, prob=0.0262 (well under the 0.30 ceiling). A later false alarm
(the 2026-07-31 30k Stage 1 scale-up's regression check) compared two Stage-1-only checkpoints
that were never expected to pass this ROI at all — see `synthetic_curriculum_plan.md`'s
correction in its Part B section. No further action needed on this ROI.

### RESOLVED (partial, postprocessing) after 3 FAILED training attempts — Stage 2 bubble/cloud halo defect (2026-07-29/30)

**The defect**: on Stage 2 bubble-fine-tuned checkpoints (starting from
`b2_full2k_finetune.pt`), a residual band of undeleted (falsely "keep") background remains
immediately around bubble/cloud contours — general (not limited to bubbles: also present, in
smaller form, on Stage-1-only cloud/glow content), and correlated with local contour
curvature (smoothly-curving contours like ovals leak more than sharp-cornered spiky/thorn
shapes). Present in small form even pre-bubble-training; substantially amplified by it.

Diagnosed via `.tmp/diagnostics/halo_diag{1,2,3,4}.py` (multi-distance ring profile 2-32px,
curvature correlation) and, critically, via `.tmp/diagnostics/real_boundary_probe.py` — a
per-angle ray-walk boundary detector built specifically because an initial hand-fit-ellipse
proxy for a real bubble's silhouette was shown (direct gray-value sampling) to be
miscalibrated, contaminating early real-instance numbers. The corrected tool walks outward
from a seed point at 24 angles through the ORIGINAL image pixels, finds the true per-angle
ink-outline transition (or flags "no-marker" if none exists within the search radius,
independent of any model), and measures delete-probability at fixed offsets past each angle's
own detected boundary — never mixing in the bubble's own interior/outline the way a
guessed-shape ring could. Applied to 6 real instances across `data/chapters-initial/001.png`/
`002.png` (1 flagged by the user + 5 gathered, deliberately mixed bordered/ambiguous); one
(inst4) was later found mis-selected (seed landed on an SFX glyph, not a bubble) and
discarded, leaving 5 valid instances.

**3 genuinely different fix mechanisms tried, all real-instance-verified (not just synthetic
aggregates), per this project's clauds (5 attempts)/black-bg (6 attempts) attempt-budget
precedent:**

1. **Data-side: curvature-weighted contour patch sampling** (`--curvature-patch-ratio`,
   `PatchDataset._curvature_coords`) — oversamples training patches centered on
   high-local-curvature contour points. **FAILED**: ring-profile re-check showed no
   improvement, moved slightly the wrong direction.

2. **Loss-side: boundary-aware loss, decoupled from `pos_weight`** (`--boundary-loss-weight`/
   `--boundary-loss-radius`, `DiceBCELoss`) — this project's model-13.0 attempt at the same
   idea had a multiplicative bug (`pos_weight` and `boundary_weight` compounded instead of
   adding, giving delete-class boundary pixels 4x the intended weight vs. keep-class); fixed
   here to compute weights additively (`class_weight = 1 + (pos_weight-1)*targets`,
   `combined = class_weight + (boundary_weight-1)`), verified as an exact no-op vs. the old
   formula when boundary weighting is off. Tested at `--boundary-loss-radius 3` (matching
   model 13.0's nominal value for comparability) and again at `16` (after establishing the
   dilation radius directly sets how many pixels around the boundary ever get reweighted, and
   3px is far short of the halo's own 2-32px extent per the ring-profile). **PARTIAL**: helped
   1 of 5 valid real instances substantially (the one ambiguous/partial-no-marker case) at
   both radii, 1 marginally (small movement only at the widest radius), 3 not at all —
   including two fully-bordered "clear control" instances with literally identical
   delete-probability at every measured band (2/4/8/16/32px), before and after, at both
   radii. Widening the radius 5x did not unlock the unresponsive instances, ruling out "radius
   too narrow" as a complete explanation (though it does explain the 2 instances that did
   move). One of the 3 unresponsive instances (`ch002` seed 350,70900) was visually confirmed
   to sit against a genuine, large true-background gutter region in a vertical-scroll page —
   not a mislabeled "keep" region — so the flat result is a real, unfixed defect, not a
   measurement artifact. Introduced a minor regression at the wider radius on the
   already-accepted-limitation `ch1_sfx_text` ROI (0.306→0.385).

3. **Data-side: background-extent-aware patch sampling** (`--background-patch-ratio`/
   `--background-min-area-frac`, `PatchDataset._background_extent_coords`) — oversamples
   boundary patches specifically where the delete-side neighbor belongs to a connected
   background component covering >= 5% of the full mask, targeting exactly the failure
   pattern mechanism 2 left unresolved ("bubble edge with a big real background area beyond
   it" is presumably under-represented in ordinary boundary-patch sampling, since most
   bubbles sit within content-dense panels). Unit-tested in isolation first (confirmed correct
   selection of the large-region case over a small sliver on a synthetic mask). **FAILED,
   more decisively than either prior attempt**: zero measurable change at every band on all 5
   real instances, including the one instance every other mechanism could move at least
   somewhat. Also regressed a previously-fixed, passing Stage 1 ROI (`ch1_diagonal_gutter`,
   this session's own Part A fix): 0.5526→0.4823, dropping below the 0.50 pass floor — a
   genuine forgetting of working content, not just an accepted-limitation drift. Checkpoint
   discarded, not adopted, per this project's standing policy that any regression on
   previously-working content stops that attempt immediately.

**Current best understanding (not fully resolved)**: real-vs-synthetic boundary sharpness/
contrast, measured with the same per-angle detector on both sources, is nearly identical
(~3% mean difference) — ruling out "real ink outlines are technically harder to see" as the
dominant driver. The pattern across all 3 attempts — every mechanism helps the single most
ambiguous/marker-poor real instance and does nothing for ordinary, unambiguously-bordered
bubbles regardless of the true background's size or clarity beyond them — is most consistent
with the model having learned a local "near a bubble body = keep" shortcut/prior that
overrides even large, textureless, unambiguous true-background regions, and that neither
loss-reweighting near the boundary (any tested radius) nor training-data oversampling (either
selection criterion tried) was able to dislodge.

**Status after the 3 training-side attempts: stopped per attempt-budget policy, not
resolved.** `b2_full2k_finetune.pt` remains the best available Stage 2 checkpoint; none of
the 3 training-side fixes are adopted.

**4th mechanism, a different class — geometric postprocessing (`--close-bubble-halo`,
2026-07-30)**: per explicit follow-up instruction, fixed geometrically on the model's OUTPUT
mask instead — the same relationship `--repair-frames` has to `--reclaim-islands`, no
retraining. New function `close_bubble_halo` (`src/ml_cleaner.py`): detects bubble/cloud
contours from the RGB's own ink outline (the same flood-fill-from-corner enclosed-hole
technique `repair_frame_interiors`/`style_analysis.extract_enclosed_holes` share, classified
via the existing `style_analysis.classify_and_measure` taxonomy) rather than eroding the
predicted mask — an erosion-based design was tried first and rejected during implementation
(the erosion depth needed to reliably strip a variable-width halo either leaves part of the
halo protected/unfixed, or is deep enough to eat real bubble content once the halo happens to
be narrower than expected on a given instance). Reclassifies keep→delete in a ring around the
detected contour only where that ring is part of the bubble's own current keep component,
within reach of a large connected delete component (true background, not a small pocket), and
not part of a "thick" region (a genuine halo is thin throughout; real adjacent content has
bulk far from any edge — computed per-pixel via distance-transform after removing every
detected bubble body, not per-connected-component, since a whole-component test wrongly drags
a thin halo down with whatever thick object it happens to touch).

Verified on the same 5 real instances with the same ring-distance methodology
(`real_boundary_probe.py`, extended with a `--close-bubble-halo` hook): closes the halo
substantially on the one ambiguous/partial-marker instance (inst1) — and in a genuinely
complementary way to the training-side fixes, improving the CLOSE bands (+2-16px) that those
barely touched — but produces zero change on the other 4 (inst2/3/5/6), same as every
training-side attempt. Root-caused directly (not left unexplained): the ink-outline detector
finds no valid bubble shape at all on those 4 instances at any tested darkness threshold
(40-140) or closing-kernel size (5-15px) — a genuine small gap in the real scan's own ink
line lets the flood-fill leak into true background, visually confirmed for one instance. This
is a different failure mechanism from the training-side one (weak boundary signal in the
source art itself, vs. a learned "near-bubble = keep" shortcut), but affects the same
instances — suggesting these specific real locations have unusually incomplete ink boundaries
that no single mechanism class was likely to fix.

Over-deletion check: zero change on every one of the 9 tracked Stage 1 ROIs (skin, steam,
sky_clouds, blank_bg, diagonal_gutter, sfx_text, caption_box) on the FULL ch001/ch002 pages.
Meanwhile the fix does change pixels elsewhere at chapter scale (27,538 px on ch001, 411,823
on ch002) — confirmed via clustering analysis to form 62 distinct, spatially-bounded groups
spread across the full ~143k-px-tall ch002 page, consistent with correctly closing many
individual bubble/cloud halos throughout the whole chapter (not just the 5 hand-picked test
instances), with zero measured risk to any tracked real-content ROI. Smoke test
(`src/smoke_close_bubble_halo.py`, mirrors `smoke_repair_frames.py`'s structure) passes all 5
required cases (halo closes; adjacent real content protected; no-op with no halo; merged
bubbles handled independently; small background pockets ignored) plus mutation/directionality
invariants.

**Status: adopted as a real, safe, PARTIAL fix, not a complete resolution.** Recommended as
an additional flag for the Stage 1+2 generation-6 checkpoints alongside
`--reclaim-islands`/`--repair-frames` (`--halo-ring-width 24 --halo-frame-darkness 40
--halo-min-bubble-area 2000 --halo-min-background-area 8000` are the defaults). Production
(`10.0-baseline.pt` + `--reclaim-islands`) stays untouched per the standing constraint. The
one identified limitation (bubbles whose own ink outline has a genuine gap in the source
scan) is specific and honestly documented, not a general failure mode. Full instance-by-
instance numbers, visual verification, and methodology detail for both the training-side
attempts and this postprocessing fix are in `notes/halo_investigation.md`.

**Combined-pipeline evaluation at full-chapter scale (2026-07-31)**: ran the full recommended
flag set on complete `data/chapters-initial/001.png`/`002.png` (not just crops). Tracked
5-instance numbers unchanged with `--reclaim-islands`/`--repair-frames` added (no
interaction). Broader 20-crop sweep across both full chapters found 2 additional bubble
instances beyond the tracked 5 closing cleanly (the fix generalizes past hand-picked cases),
no new failure mode from combining all 3 postprocessing flags together for the first time,
and correctly-behaving full-bleed splash panels/gutters/diagonal edges throughout. One
relevant-but-out-of-scope observation: a small halo variant on an unenclosed SFX glyph stroke
(`close_bubble_halo` only detects enclosed ink-outline shapes, so never reaches this) — noted
for the Stage 3 SFX work below, not fixed here. **Stage 2 (with postprocessing) confirmed
genuinely done.**

**4th mechanism attempt, gated on a real diagnostic (2026-07-31)**: before attempting
anything, ruled out the auxiliary SDT head as a candidate 4th mechanism — it's the same
"boundary emphasis during training" class as this week's Attempt 2 and historically clauds'
models 13.0/15.0, both already tried and already the subject of this doc's own explicit
warning against a further lever in that class. Not attempted. The architectural/receptive-
field hypothesis (all 4 guidance channels are local-only, ~200-220px computed receptive
field) was undiagnosed, so before deciding whether to implement anything, built a bounded
occlusion probe (`.tmp/diagnostics/halo_context_occlusion_probe.py`): on 2 unresponsive real
instances, progressively occluded everything beyond radius 32/64/128/256px (and no
occlusion) from the seed, re-ran raw inference, measured delete-probability at the halo. All
occlusion levels — including full context — gave exactly 0.0000 delete-probability at every
band, on both instances: the model already confidently predicts "keep" with almost no visible
surroundings, so more context (up to and past its own receptive field) changes nothing. This
directly falsifies "insufficient context" as the driver, further confirming the "near-bubble
= keep" local-prior/shortcut explanation from a genuinely different angle (behavioral
occlusion, not another training-attempt outcome). **No genuinely new, actionable training-
side mechanism identified** — 4 independent approaches (3 training-side + this diagnostic)
now confirm the hard wall. `--close-bubble-halo` remains the standing solution; no further
training-side attempts planned without new evidence.

### RESULT (quick_smoke-verified, scale-up deferred) — Stage 3 hollow-shape SFX-cleanup hypothesis (2026-07-31)

With Stage 2 (training + `--close-bubble-halo` postprocessing) confirmed genuinely done above,
implemented the first of the two queued Stage 3 hypotheses
(`notes/stage3_sfx_hypotheses.md`): hollow shapes (oval/square/trapezoid outline, plain
interior) as an explicit training signal that an enclosed shape's interior is not always
"keep" the way a bubble's is. The note's own precondition was explicit: a bare hollow shape
would reproduce the skin_neck shortcut (ambiguous soft/light interior inside a bounded
outline), so the implementation required a real structural differentiator, not just
brightness/hue.

**Implementation** (`PepperNCarrotDataset/src/synthesize/synthesize_curriculum.py`):
`fill_sfx_hollow_interior` (a flat, near-white fill respecting `LIGHT_FILL_CEILING`),
`render_hollow_sfx_shape` (PIL-drawn oval/square/trapezoid, border thickness 15-30% of the
shape's own half-size — an order of magnitude thicker than a bubble outline's 1.5-6.5%, and
never any text in the interior, unlike bubbles which always have `render_bubble_text` called
on them — both structural, guidance-channel-visible cues, not textural ones), and
`composite_hollow_sfx` (reuses `place_and_paste`'s placement/alpha-compositing, then
explicitly corrects the interior back to delete=False afterward using the known shape
geometry, since `place_and_paste` itself unconditionally marks opaque pixels keep=True and
was not modified). Wired into `generate_sfx_page` alongside the existing asset-based
`composite_sfx` path (unchanged), at a 45%-per-page chance of one instance, deliberately rare
since this is a targeted augmentation, not dominant Stage 3 content.

**Verification, full ladder**: `preflight_check.py --new-kind sfx_hollow_interior` —
initially failed the contrast-band check (fill sampled up to 250, exceeding
`LIGHT_FILL_CEILING=240`; fixed to sample `[200, 238]`), then all 3 checks passed clean,
including the statistical-overlap check that was expected to flag risk — it didn't, this
fill's brightness/hue turned out separable enough on its own. `run_ladder.sh` quick_smoke
(250-tier, `--resume b2_full2k_finetune.pt --lr 3e-5`) trained cleanly, no NaN, best
checkpoint val_loss 0.092. `regression_suite.py --pairwise` against the same
`b2_full2k_finetune.pt` baseline: **26/27 checks passed** — the 1 failure is
`ch1_sfx_text`, an already-accepted pre-existing Stage 1 limitation (see its own entry above),
drifting modestly further (+0.029), not a new regression from this feature. **All 10 pairwise
gaps passed**, including `sfx_vs_skin` (gap +0.027, well under the 0.15 threshold) — the
specific skin_neck-shortcut signature this hypothesis was built to avoid was not reproduced.
A targeted real-instance spot-check (a genuine styled SFX glyph in `ch001` with a solid white
fill and thick dark outline — structurally similar to the synthetic training pattern, and
exactly the kind of case that should stay fully "keep," not be treated as a hollow window)
showed identical (fully-keep) behavior on both the pre- and post-quick_smoke checkpoints — no
degradation on the real case this hypothesis's own risk was about.

**Status: quick_smoke-verified, clean; 2k/10k scale-up deliberately deferred, not attempted.**
This is a conservative stopping point given the scope already covered this session (halo
investigation, full-chapter pipeline evaluation, the Part 2 diagnostic, and this Stage 3 work
all in one day), not a finding of any problem — mirrors the existing "B3: decide on 10k Stage
2 scale-up" open item's own precedent of leaving a scale-up decision as an explicit, separate
step rather than auto-escalating. Hypothesis 2 (SFX-outline-merges-with-frame-border) remains
untouched, still an open question per its own framing in `stage3_sfx_hypotheses.md`.

## Halo defect investigation: CLOSED (2026-08-01), 5 mechanisms tried and discarded

Full detail throughout `notes/halo_investigation.md`. Summary for future reference: the
Stage 2 bubble-fine-tune halo defect (undeleted background band, 2-32px, around bubble/cloud
contours, curvature-correlated) was investigated across **5 independent mechanisms**, spanning
data-side, loss-side, architecture-side, and (twice) an independent-refiner approach:

1. **Curvature-weighted contour patch sampling** (data-side oversampling) — failed, no
   measurable effect.
2. **Boundary-aware loss reweighting**, decoupled from `pos_weight`, tested at 2 radii
   (loss-side) — helped exactly 1 of 5 real tracked instances (the most ambiguous one), zero
   effect on the other 4.
3. **Background-extent-aware patch sampling** (data-side, different oversampling criterion) —
   zero effect, plus a real regression on an already-fixed Stage 1 defect.
4. **An auxiliary SDT head** — ruled out by research without implementation: same "concentrate
   gradient at boundary pixels" mechanism class as 3 *earlier* independently-implemented
   mechanisms (capacity increase, boundary-weighted BCE, an SDT head, model 18.0's in-trunk
   `RefineHead`) that all made boundary precision *worse* on this same small `SmallUNet` for a
   related defect ("clauds") — a converged, repeated signal that concentrating extra gradient
   pressure at boundaries pushes past this architecture's stability limit rather than through
   it. A bounded occlusion probe additionally falsified "the model needs more context" as an
   explanation: delete-probability at the halo location was exactly 0.0000 at every context
   radius tested (32/64/128/256px, full context) — a context-independent, saturated local
   prior, not a receptive-field limitation.
5. **A small, from-scratch, trunk-independent crop-based refiner** (CascadePSP-inspired: keep
   the one thing that ever meaningfully helped a related defect — an independently-trained
   corrector, not a trunk modification — while fixing CascadePSP's two disqualifiers,
   third-party-weight licensing and full-page inference cost). Tried in its two most natural
   variants:
   - **Synthetic perturbation-based training** (calibrated outward-ring delete→keep flips,
     severity/width randomized to match the measured ring-distance profile): the trained
     refiner reversed its own perturbation function's statistical signature with 95-100%
     accuracy on held-out same-distribution crops, but showed **zero transfer** to a
     differently-constructed synthetic test or any of the 5 real tracked instances — it had
     learned to undo a specific noise texture, not the real defect (confirmed not a bug via
     direct in-distribution verification).
   - **Real-error-based training** (the b2 checkpoint's own predicted errors on its own
     synthetic distribution vs. clean ground truth — verified the halo pattern genuinely
     appears there first: mean predicted delete-fraction 0.014/0.068/0.368/0.594/0.822 at
     +2/+4/+8/+16/+32px across 112 real bubble instances, the same qualitative shape as real
     instances). This removed the "wrong perturbation proxy" confound entirely, and showed
     healthier training dynamics (gradual loss decrease, no near-zero collapse, a flip-ratio
     signature opposite to model 18.0's rather than matching it) — **but still zero transfer**
     to the same synthetic test or any real tracked instance.

   Two data-construction approaches with fundamentally different motivations and failure
   modes converged on the identical outcome — strong evidence this is a generalization gap
   intrinsic to the mechanism (a small, from-scratch, crop-scoped network trained only on this
   project's synthetic bubble renderings), not a fixable data-proxy problem.

   Separately, a cheap prerequisite check (gap-tolerant `--close-bubble-halo` contour closure,
   widening the stroke-closure kernel up to 35x35) found zero benefit at any tested size, and
   precisely diagnosed *why* the one remaining real failure case (inst3) can't be fixed
   geometrically: its ink outline is fully closed, but the tail touches an unrelated
   panel-divider line, merging two structurally different objects into one connected component
   — a semantic distinction ("is this connection part of the same object?") no amount of
   morphological closing can make.

**Verdict: hard wall confirmed across data-side, loss-side, auxiliary-head-side, and
independent-refiner-side mechanisms (5 total, one with 2 sub-variants) — mirroring this
project's own "clauds" precedent of stopping after repeated convergent failure across
genuinely different approaches. `--close-bubble-halo` remains the standing, accepted
solution** (helps ~1 in 5-6 real instances via hand-coded ink-outline flood-fill geometry, a
known, bounded, safe fix). No further halo mechanism attempts without genuinely new evidence.

### Reopened and re-verified at full-page scale (2026-08-02) — same verdict, strengthened

The closure above was reached using `real_boundary_probe.py`'s default 600×600 windowed crop
around each seed point, not full-page (production) scale. Direct re-verification found this
had already produced one wrong conclusion (inst3's "structurally undetectable" diagnosis was a
crop-scale artifact — at full-page scale it IS detected and `close_bubble_halo` DOES help it).
Triggered a full re-verification (`notes/halo_investigation.md`, Part A): all 5 instances
re-measured at full-page scale (3/5 remain genuine zero-halo controls, confirmed), plus 2
additional independent variables tested on the 5th mechanism that hadn't been isolated before —
**crop-size** (inference-only 224→512, and a proper 512 retrain — both zero transfer) and
**training-corpus diversity** (fresh 600-image pool, different seed — zero transfer, even more
completely than every prior variant: every band, every instance, exactly 0.0000). A geometric
alternative (Stage 1 + bubble-detector force-keep-interior, bypassing the learned refiner
entirely) was also tried: moderate but incomplete real-world recall (38-64% across 3
manually-cleaned reference chapters) plus an unquantified false-positive risk, not adopted.
**Verdict unchanged, now on stronger footing**: 4 independent variants of the crop-based
refiner (2 data sources × crop-size × diversity) plus a structurally different geometric
approach all converge on the same answer. Full detail, all real numbers, in
`halo_investigation.md`'s "CORRECTION" section. A full CascadePSP-style global+local redesign
(Part A.6) remains a legitimate untried direction, deliberately not started given the
overnight time budget — not discarded.

## Instance-aware architecture pivot, Part 0-2 (2026-08-03) — SFX proof-of-mechanism positive (n=6), calibration gap found and resolved on the 3rd attempt

Full detail: `notes/instance_aware_pivot_2026-08-03.md`. Time-boxed session (requested scope:
full detect-then-segment instance-aware pipeline for frames+bubbles+SFX). Part 0 (code-reading
only, no training) corrected the premise before executing: the only existing real-image
object-proposal detector (ink-outline flood-fill, `extract_enclosed_holes`/`close_bubble_halo`)
only proposes a valid box on **1 of the 5** halo-investigation tracked instances — wherever it
works, the boundary problem is already solved for free (that's what `close_bubble_halo` does),
so an instance-aware model only adds value on the 4/5 already-exhausted hard cases. **Scope
narrowed to SFX specifically**: no existing geometric detector, and `ch1_sfx_text` is a real,
open, well-diagnosed defect (30.7%→19.9% across the training chain, never closed) with a
positive, fixable occlusion-probe signature (correct at R<=64px, flips at R>=128px) — unlike
halo's flat 0.0000-everywhere hard wall. Chose a dense **local**-crop mask (96px margin,
inside the occlusion probe's "still correct" band) over a full parametric contour, explicitly
because of the time budget, not a claim contour is worse (this project's history — models
12.0/13.0/15.0 — shows under-verified training mechanisms produce misleading results, and a
rushed Deep-Snake implementation couldn't get real verification today).

**Result: positive, real n=6 evidence, not an anecdote.** A tiny (~1/10th SmallUNet width)
from-scratch local-crop network, trained on 1146 crops extracted from the existing
`stage3_sfx_2k` pool (new ink-stroke connected-component instance detector,
`src/research/build_sfx_instance_crops.py` — no `PepperNCarrotDataset` changes), evaluated
against the tracked `ch1_sfx_text` instance plus 5 additional real SFX-like instances found on
the manual-reference chapters and visually confirmed one by one: **6/6 PASS** (mean delete-prob
0.03-0.22 vs. the 0.30 ceiling), every one beating the best whole-page checkpoint on record for
`ch1_sfx_text` (0.0935 vs. `stage3_sfx_2k_resumed`'s 0.1987). Consistent with the mechanism:
bounding the crop to 96px structurally prevents the R>=128px trigger from ever being visible.

**A follow-up calibration fix regressed the metric that actually mattered — a real, reproduced
finding, not noise.** The smoke checkpoint under-predicted delete-confidence on pure blank
crops (0.20, technically correct side of 0.5 but weak — traced to 46% of glyph-centered
training crops having zero delete pixels at all). Adding background-only negative crops fixed
this cleanly (0.99) but **regressed all 6 real instances from PASS to FAIL** (0.43-0.69) —
reproduced exactly on a second from-scratch run with a clean regenerated dataset. Same
conservatism/aggression-pendulum shape as this project's larger-scale precedents (black-bg
dilution attempts, models 12.0/13.0/15.0's boundary-emphasis regressions): naively adding more
of an underrepresented case overcorrected instead of balancing. **Not adopted** — the
`glyph_only` checkpoint remains the standing result; both checkpoints kept on disk
(`.tmp/checkpoints/instance_sfx_smoke/`), not overwritten, so the comparison stays
reproducible.

**Follow-up: ratio-tuning and post-hoc recalibration both also fail to fix this — closed, not
just untried.** A lighter background-crop mixture (25% of pages, not 100%) still regressed all
6 real instances (mean_prob 0.36-0.60, still over the 0.30 ceiling) despite also fixing
blank-crop calibration (0.91) — ruling out "just use less" as a fix. A cheap post-hoc global
logit-bias sweep (no retraining, [-6,+6]) found no single value where both blank-crop and all
6 real instances pass simultaneously — the crossover points for each requirement land at
almost the same bias, meaning the `with_bg` family didn't just shift, it **lost the
discriminative margin** between real SFX-glyph content and blank background relative to
`glyph_only`. This rules out both data-ratio tuning and output recalibration as fixes for this
sub-problem.

**Third attempt at the calibration sub-problem — SUCCESS.** Same full background-crop data as
`with_bg`, but background-only crops down-weighted to 0.2x in the loss (`dice_bce`'s new
`sample_weight` argument, verified an exact no-op when unset — every other variant's behavior
unaffected) instead of adjusting data quantity or output bias. Result: **6/6 real instances
PASS** (mean_prob 0.09-0.29, `real_cand_4` closest to the ceiling at 0.289) **and** blank-crop
calibration fixed (0.586, confidently delete) — the first variant to jointly satisfy both
properties. Confirms the fix needed to happen in the loss function (letting the model see
background examples without their gradient dominating as heavily as instance examples), not
in the data mixture or a post-hoc output shift. Per this project's attempt-discipline
convention, stopping the calibration side-investigation here because it succeeded, not
because the budget ran out. `with_bg_weighted` is now the more complete of the two checkpoints
kept from this session (`glyph_only` remains the cleaner single-variable reference result).

**Explicitly NOT production-ready**: no real (non-heuristic) SFX object-proposal detector, no
integration logic with the whole-page dense model, no regression-suite battery (network never
saw skin/steam/bubble content). The calibration caveat that was open at the end of Part 2 is
now resolved; these three remain the actual blockers to any real use.

## Instance-aware architecture pivot, Part 1 (2026-08-04) — real object-proposal detector + whole-page pipeline

Full detail: `notes/instance_aware_pivot_2026-08-04.md`. Closes the biggest blocker from the
2026-08-03 session: `with_bg_weighted` had only ever been tested on pre-cut crops with
hand-specified boxes, never on a whole page where objects must first be found. Built
`find_sfx_instances()`'s first real precision/recall measurement (a stratified real-page sample,
not hand-picked candidates): raw SFX-specific precision is low (~20-25%, not cheaply fixable by
geometric threshold tuning — no clean separator found between SFX/bubbles/captions/noise), but a
direct dense-baseline comparison showed the false-positive rate poses no functional regression
risk (every false-positive candidate's bbox is already predicted correctly by the existing dense
checkpoint, so the correction is a no-op there) — a real, measured mitigation, not an assumption.

Built the whole-page pipeline (`src/research/sfx_instance_pipeline.py`, mirroring
`apply_halo_refine`'s detect→crop→model→paste-back shape) and ran it end-to-end on real chapters
(`eval_sfx_pipeline_e2e.py`): **6/6 recall** on the known real instance set via unconstrained
whole-page detection (not hand-specified boxes). Per-instance comparison against the dense
baseline was a mixed but still-passing picture (2 improved, 2 slightly regressed, 2 unchanged;
all 6 still under the 0.30 ceiling) — reported honestly rather than rounded up to a clean win.
This is the first real instance-aware production *candidate* for SFX specifically, still gated
on CLI integration and a real regression battery before any deployment consideration.

## Instance-aware architecture pivot, Part 2 (2026-08-04) — class-generalization test on halo: no transfer, confirms the closed investigation rather than reopening it

Full detail: `notes/instance_aware_pivot_2026-08-04.md`. Tested whether the SFX mechanism (local
crop + 0.2x-weighted background-loss training) generalizes to the original motivating defect,
bubble/cloud halo — using the existing bubble contour detector (`_find_bubble_interior_holes`,
already validated, nothing new needed there), the same `TinyInstanceNet` architecture, and the
exact same `real_boundary_probe.py` ring-distance methodology used for `HaloRefinerNet`'s 4-variant
evaluation (the halo investigation's already-closed 5th mechanism), for direct comparability.

**Result: no transfer, and two small new regressions on the 5 tracked real instances** — inst1's
+32px band roughly doubled (0.091→0.236), inst6 gained small new non-zero values where baseline
was clean 0.0, inst3 (the topologically-hard tail-touches-panel-divider case) stayed exactly
unchanged, inst2/inst5 (zero-halo controls) stayed unaffected. Not an ambiguous or broken result
(training was clean, output non-degenerate) — a clean, interpretable non-transfer, exactly
consistent with the halo closure's own occlusion-probe finding (0.0000 sensitivity at every
context radius, i.e. context-*independent*) vs. SFX's context-*dependent* signature that made
crop-bounding actually work there. **Confirms, with a mechanistically distinct new architecture
(fresh local prediction from guidance channels, not `HaloRefinerNet`'s mask-refinement approach),
that halo's hard-wall is real and general, not an artifact of the specific designs tried before.**
`--close-bubble-halo` remains the standing solution. Halo investigation's closed status stands,
reinforced rather than reopened — no further halo-mechanism attempts without genuinely new
evidence.

## Halo attempt 7 (2026-08-04/05): Deep Snake-style parametric contour deformation — CLOSED (2026-08-04 17:57:24 EEST), targeted fix verified sound but negative on the primary metric

Full detail: `notes/instance_aware_pivot_2026-08-04.md`. A genuinely new mechanism class, not a
variant of the 6 already-tried ones — every prior mechanism (5 dense whole-page + 1 dense
instance-scoped crop) still produced a dense per-pixel probability field somewhere in its output;
this predicts/deforms a sequence of boundary vertices directly (a `TinyInstanceNet`-style CNN
backbone + circular Conv1d "graph conv" over the closed contour cycle, predicting one radial
offset per vertex from a coarse initial ellipse), removing the substrate the halo occlusion probe
showed a context-independent "keep" prior can leak across.

**Two real bugs found, root-caused, and fixed along the way** (not guessed at, each confirmed by
direct measurement before being called a bug): (1) the synthetic training crop size was
accidentally truncated to the smaller half of the real bubble-size distribution (caught by a
visual spot-check, not just trusting an encouraging number — crop size 320→512px fixed it), and
(2) two training runs died with zero output from a silent OOM (unbatched validation at the larger
crop size, compounded by fully-buffered stdout hiding all progress) — fixed via batched
validation + unbuffered logging. Both fixes are durable improvements to this project's evaluation
infrastructure regardless of this attempt's ultimate verdict.

**Escalating 250→1k→2k real-instance checks against the same 5 tracked halo instances (inst1/2/3/
5/6) initially looked like a clean, improving trend culminating in a 5/5-improved result at 2k —
until a visual check (per this project's own standing "spot-check before trusting a number" rule)
caught a THIRD bug: the ray-walk ground-truth technique (`real_boundary_probe.py`'s per-angle
ink-darkness walk, previously validated only for dense-model probability measurements) was
frequently catching interior TEXT ink as a false "boundary," not the true, much larger bubble
outline — confirmed directly on inst3, where every one of 64 angles returned a 1px "boundary"
under the original settings (the seed point sits essentially inside a text glyph). Fixed
(`min_run` 2→5, `max_radius` 220→390, a `MIN_TRUSTED_RADIUS=15` filter) and re-measured all three
scales.**

**Honest result under the corrected metric: 250=2/5 improved, 1k=4/5 improved, 2k=2/5 improved —
non-monotonic, not a clean scaling trend**, with deltas mostly small (1-10px) relative to the
much-larger-than-previously-reported true error scale (26-103px). This pattern reads as
measurement noise at an underpowered real-instance sample size (n=5), not a reliable signal.
Regression check (v2_scale1k, the most consistent checkpoint): 8/8 ROIs unchanged, no
regressions — safe, just not clearly effective. **Verdict: PARTIAL/INCONCLUSIVE — not declared
working (data doesn't support it) and not declared a clean architectural dead end either (nothing
here shows the hard 0.0000-at-every-radius wall the original 6 mechanisms hit; synthetic learning
is genuinely strong and clean throughout).** A legitimate third category alongside "worked" and
"failed" that this project's own history already makes room for. `--close-bubble-halo` remains
the standing production solution; this mechanism is not adopted. If revisited, a larger
real-instance sample (not just more training data at n=5 eval instances) is the most promising
next lever, ahead of a further architecture change.

### Part A follow-up (2026-08-04 17:07:32 EEST): manual-clean metric promoted to primary, reveals gen-6 combined checkpoint trails production on full-pipeline accuracy

`eval_gen6_checkpoint.py` extended to accept an arbitrary postprocess chain and generalized to a
pixel-weighted multi-chapter aggregate; this manual-clean comparison (over-del%/under-del%/total
pixel error against human-cleaned reference chapters) replaces the 5-instance ring-distance check
as the PRIMARY halo-mechanism metric (ring-distance stays secondary/diagnostic). Chapter 035 was
attempted but excluded — its reference pair has a genuine 480px height misalignment, a
newly-discovered data-quality gap, not a metric bug; GT set is 001/002.

Fresh numbers: `10.0-baseline.pt`+islands (production) = 13.0241% aggregate total error;
gen6-combined (Stage1+2+3) + full chain (islands→repair-frames→close-bubble-halo) = 16.6585%.
**The gen-6 combined checkpoint + full chain currently measures worse than production on this
metric**, almost entirely from higher under-deletion (~16.2% vs. production's ~12.4%) rather than
over-deletion (gen-6 is actually better there: ~0.44% vs. ~0.59%) — i.e. gen-6's base checkpoint is
more conservative about what to delete, and that costs more accuracy than the halo-closing chain
recovers. This is a full-pipeline number, not an isolated halo-mechanism measurement: the chain
operates on top of whatever the base checkpoint already predicted, so future attempt evaluations
should track a given checkpoint's own raw-vs-chained delta under this metric, not the chain's
absolute number against a different (weaker) base model.

### Part B follow-up (2026-08-04 17:57:24 EEST): targeted fix verified sound, attempt 7 CLOSED — negative on the primary metric

Added a cosine LR schedule + best-val-loss checkpoint tracking to `contour_deform_net.py`,
directly targeting the diagnosed cause of attempt 7's non-monotonic real-instance results (flat
LR, final-epoch-only saving). **The fix works as diagnosed**: at both smoke (250) and 1k scale,
retrained cleanly with best-epoch == final-epoch (val_loss monotonically improving under the
schedule, no late-run degradation) — confirms the training procedure itself is no longer the
suspect.

Verified the resulting `v3_scale1k` checkpoint with both metrics. Ring-distance (secondary):
4/5 real instances improved, same ratio as before the fix. **Manual-clean (primary), applied via
`contour_instance_pipeline.py` on top of gen6-combined's full chain: aggregate total error goes
from 16.6585% (chain alone) to 17.3112% (chain + contour refine) — a clean, both-chapters
regression**, driven by over-deletion nearly tripling (0.44%→1.24%) while under-deletion barely
improves (16.22%→16.07%). **This is the exact scenario Part A's metric upgrade was built to
catch**: a mechanism reading as "4/5 improved" on the isolated boundary-point ring-distance check
is a net real-world accuracy loss once measured end-to-end against actual manual-clean pages —
ring-distance doesn't see that the deformation pushes outward into content that should stay kept.

**Verdict: attempt 7 (Deep Snake-style parametric contour deformation) CLOSED, not adopted.**
Training-procedure issues are ruled out as the explanation (the fix demonstrably worked and made
it worse, not better); the weak/negative real-instance transfer is a property of the mechanism or
its training signal itself. `--close-bubble-halo` remains the standing production solution. Per
plan, proceeds to attempt 8 (Part C).

### New finding (2026-08-04 20:54:35 EEST): SFX under-protection pattern is a SEPARATE defect from bubble-halo, not the same mechanism

Unified-mechanism pre-check (`.claude/plans/snazzy-cuddling-creek.md`), run before committing to
CRF's 1k-scale verification: does a newly-observed SFX "under-protection" pattern (background left
undeleted around a real SFX glyph, current production `10.0-baseline.pt`+islands) — distinct from
the already-tracked, opposite-direction `ch1_sfx_text` OVER-deletion issue (2026-07-29/2026-08-03)
— share bubble-halo's already-characterized context-independent, saturated occlusion signature?

Two real instances confirmed visually first (both within the already-tracked `ch1_sfx_text`/
`ch1_blank_bg` ROI area, `data/chapters-initial/001.png` — see
`.tmp/diagnostics/sfx_underprotection_candidates/final_two_instances.png`): a clear gray
(kept/undeleted) halo bleeding beyond each glyph's own ink strokes into surrounding blank area that
`ch1_blank_bg` establishes should be deleted. Ran the SAME occlusion-probe methodology bubble-halo's
original diagnosis used (`.tmp/diagnostics/halo_context_occlusion_probe.py`'s `occlude_beyond`/
`measure_ring_bands`, same RADII ladder, same pre-committed >0.05 sensitivity threshold) —
boundary geometry sourced from each glyph's own ink contour (`cv2.findContours`), not the
bubble-calibrated ray-walk (`find_per_angle_boundary` still has its old, un-patched
text-contamination-prone defaults — deliberately not reused here for exactly that reason).

**Result: both instances are clearly CONTEXT-DEPENDENT, not saturated.** `glyph_A_Yu`: sensitivity
0.0504/0.1304/0.2817 at +8/+16/+32px (all >0.05). `glyph_B_O_stroke`: 0.0350/0.1967/0.2500 (2 of 3
bands >0.05, max well above threshold). In both cases delete-probability near the glyph rises
substantially as more context becomes available (32/64px occlusion ≈0, 256px/full context up to
0.28) — the opposite of bubble-halo's exactly-0.0000-everywhere-regardless-of-context signature
(`notes/halo_investigation.md`, inst2/inst5, zero sensitivity at every radius).

**Verdict: this is a separate, distinct defect from bubble-halo, not the same underlying
mechanism.** The unified-mechanism hypothesis is not supported by this evidence — SFX
under-protection behaves like the *already-known* `ch1_sfx_text` family (context-dependent,
correct with less context / degrades with more — the same qualitative shape that made the SFX
instance-scoped model's local-crop-bounding approach work for the over-deletion direction), not
like halo's context-independent local prior. Does not block or redirect CRF's evaluation, which
proceeds against bubble-halo only, as already scoped. Worth a separate future look (not today):
whether the existing SFX instance-scoped architecture (`instance_sfx_net.py`) could also help this
under-protection direction, given the shared context-dependent signature with the issue it already
fixes — noted as a lead, not pursued this session.

### Attempt 8 (CRF) follow-up (2026-08-05 05:01:12 EEST): 1k real-instance verification CLOSED — near-zero real engagement, decisively worse than attempt 7's pattern

Verified the 1k CRF checkpoint with both metrics. Ring-distance: `crf_refine` alone byte-identical
to no-postprocessing baseline on all 5 tracked instances; a direct pixel-diff on the one instance
with a correctly-detected hole showed CRF touching only 0.027% of the crop, missing the measured
halo ring entirely. Manual-clean (primary, unaffected by the harness issue below since it runs
CRF's own full-page hole detection per chapter): 16.6252% vs. 16.6585% baseline, a noise-level
0.033pp change. **Verdict: CLOSED, not adopted** — more decisively negative than attempt 7 (near-
total non-engagement with real content, not just weak/inconsistent transfer), confirmed two
independent ways.

**Real, durable methodology finding, independent of CRF's own fate**: `_find_bubble_interior_holes`
(shared by `apply_halo_refine`, mechanism 5's original real-instance harness, and the new
`apply_crf_refine`) found zero holes on 2/5 tracked instances and a hole not containing the seed
on 2/5 more — only 1/5 correctly targeted. Since mechanism 5 (HaloRefinerNet) used the identical
detection call in its own original real-instance evaluation, this same gap may have affected that
closure too — worth remembering if mechanism 5 is ever revisited, though not itself grounds to
reopen it without new evidence.

Also tested (zero new training) whether the already-proven SFX instance-scoped refiner
(`with_bg_weighted`, 6/6 real transfer on the *opposite*-direction over-deletion problem) helps
the new under-protection pattern found earlier tonight: engaged meaningfully more than CRF
(5.4%/3.8% of crop pixels touched) but didn't visually close the halo, and moved the full-chain
manual-clean aggregate by less than noise (16.6790% vs. 16.6585%, slightly worse if anything).
Read as a crop-margin mismatch (that model's ~96px training margin plausibly doesn't reach the
100px+ halo extent seen tonight) and a metric-sensitivity limit (a narrow, real, localized fix is
invisible against a whole-chapter aggregate), not as new evidence against that model's own
already-established real result.

**No overnight training launched tonight** — per this project's own "don't manufacture a fix
without real evidence for what's wrong" discipline, neither result above points at a specific,
diagnosable fix worth committing unattended compute to blind. A properly-scoped attempt 9 (or a
redesigned, wider-margin under-protection-specific SFX training task) is next-session work.

## MISSION: near-manual-clean staged inference pipeline (2026-08-05)

New mission, planned by Fable: geometric/staged inference on the pure Stage 1 checkpoint
(`18.0-frames.pt`), not further learned halo mechanisms. Full plan:
`.claude/plans/snazzy-cuddling-creek.md`. Adoption bar: ≤5.0% aggregate manual-clean total error,
over-del ≤1.5%, both numeric and visual confirmation required.

### Phase 0b (2026-08-05 07:42:20 EEST): chapter 035 GT "misalignment" was never real — fixed, added to the harness

Prior exclusion (coarse top/bottom-shift probe) was wrong. Masked kept-pixel-only comparison at
shift=0 across all 162,376 valid rows: 99.99% match with near-zero diff (mean 0.23/255) — the
cleaned reference is a clean prefix (`035.png[:162376]`), not a misalignment; the extra 480 rows
are a real, correctly-un-cleaned promo/credits page (visually confirmed). Fixed via
`GT_HEIGHT_OVERRIDE` in `eval_gen6_checkpoint.py`; 035 now in `GT_CHAPTERS`.

Fresh 3-chapter baselines: production 21.25% aggregate, gen6-combined+full 24.03%, **Stage1 pure
(`18.0-frames`+islands) 20.19%** (still best). 035 alone is far harder than 001/002 (~33%
total error for every config tested) but the SAME dominant failure mode: 89.5% of its
under-deletion is near-black backdrop content, confirming Phase 1's priority rather than
changing it. The mission's real starting point is 20.19%, not the 11.40% 2-chapter figure used
when the plan was first written.

### Phase 1 (2026-08-05 08:04:58 EEST): geometric black-backdrop reclaim — 3 discriminators tried, none generalizes, stopped for reassessment

`reclaim_black_backdrop` (large near-black full-width components → delete) looked strong in
total-error terms (Stage1+islands+backdrop aggregate: 20.19%→10.69%) but over-deletion exploded
1.07%→8.24%, over the ≤1.5% adoption cap. Root cause confirmed visually: real dark ART panels
(e.g. a stylized game-HUD splash panel with lightning/UI boxes) get misclassified as gutter.
Three discriminators tested against real per-component GT-precision, not anecdote — per-component
grayscale std, enclosed-bright-hole fraction, and color chroma — the third looked perfectly
separated on chapter 001 alone but failed on 002/035 (035 shows continuous, non-bimodal
GT-precision within single connected components, meaning many real components are genuine
gutter/content mixtures that no whole-component threshold can correctly split). Full detail:
`notes/instance_aware_pivot_2026-08-04.md`. Stopped per the plan's own 3-iteration gate rather
than shipping a chapter-001-overfit rule.

### Phase 1 v2, Gate 1a (2026-08-05 08:26:01 EEST): model has zero per-pixel signal on black content — same saturated-prior signature as bubble-halo

Pre-committed probe (`.tmp/diagnostics/backdrop_prob_probe.py`): does `18.0-frames.pt`'s own
continuous delete-probability separate true gutter-black from real dark-art-black? Recall stayed
at 0.0000-0.0021 across the ENTIRE threshold sweep (0.05-0.95) — the model essentially never
predicts meaningful delete-probability on near-black pixels regardless of threshold. Same
context-independent saturated "confident keep" pattern documented for bubble-halo's original
occlusion probe, now confirmed on a second, unrelated defect class. FAIL by a wide margin against
the pre-committed rule; per plan, no iteration — moved straight to the training-data fallback.
Confirms the near-black class is 78.7-89.8% of Stage1's total under-deletion across all 3
chapters, validating it as the dominant lever the mission's numeric target depends on.

### Gate 1c, Attempt 1 (2026-08-05 09:27:43 EEST): black-backdrop training-data fallback shows REAL transfer, with a diagnosed, fixable flaw

Recolored GT-delete-labeled background from white to near-black in 250 of 500 mixed Stage-1
training pages (`src/research/make_black_backdrop_variant.py`, no sibling-repo edits), resumed a
5-epoch finetune from `18.0-frames.pt`. Manual-clean 3-chapter result: aggregate 20.19%→14.69%,
under-deletion down 8.1pp — **real transfer to real chapters, unlike attempts 7/8's near-zero
result.** Over-deletion also rose 1.07%→3.71%: decomposition traced this to the canonical
HUD-splash-panel case (the same one that broke Phase 1 v1's geometric approach) — the model now
correctly protects the tight margin around embedded UI boxes but over-generalizes the surrounding
panel background to "black → delete" anyway. Root cause: every training example's darkened
region was GT-delete-labeled; zero contrastive examples of dark-but-real-content taught the model
no basis to avoid a pure-darkness shortcut. Iterating (Attempt 2/3): add a third variant type that
darkens (preserving texture, not flat recolor) real GT-keep content, so the model sees both
directions.

### Gate 1c, Attempt 2 (2026-08-05 12:43:28 EEST): contrastive-variant fix REGRESSED past the original baseline — stopped, not iterating blind

Added a `_darkcontent` variant (darkens GT-keep content, preserves texture) to fix Attempt 1's
over-generalization. Result: aggregate 25.39% (over 4.14%/under 21.25%) — worse than Attempt 1
(14.69%) AND worse than the untouched `18.0-frames`+islands baseline (20.19%) on both axes.
Notably, Attempt 2's own synthetic val_loss (0.190) was BETTER than Attempt 1's (0.270) — no
correlation with real transfer direction, a third reinforcement of "synthetic metrics gate, never
prove." Stopped the training-data ladder per the plan's own gate rather than guess at a 3rd
variant; `blackbg_v1.pt` (14.69%, real transfer confirmed, diagnosed-but-unresolved over-deletion
flaw) remains the best result this mission has produced, not adopted. Next attempt needs a fresh
decomposition before any further training design, not another blind variant tweak.

### MISSION PLAN v3, Step 1 (2026-08-05 13:54:52 EEST): confidence-gated fix on v1 — clean FAIL, no attempt budget spent

Tested whether `blackbg_v1.pt`'s own continuous probability (now trained on the black-backdrop
distinction, unlike `18.0-frames.pt`'s zero signal at Gate 1a) could fix v1's HUD-panel
over-deletion at inference time alone: geometry gates WHERE (near-black candidate population),
probability gates WHICH (raise the decision bar only there). Real signal now exists (76.9%
precision/86.2% recall pooled at tau=0.05, vs. the old model's sub-0.21%-recall everywhere), and
the canonical HUD-panel case is dramatically fixable in isolation (27.66%→0.28% wrong at
tau=0.90). **But it doesn't generalize**: real projected outcome at every tested tau shows
aggregate total 23.88-24.28%, worse than v1-as-is (14.69%) — raising the bar broadly also rejects
a large share of TRUE gutter pixels sharing the same borderline confidence band, so
under-deletion explodes (10.98%→20.9-21.5%) far more than over-deletion improves. Decisive,
cheap, real negative result — no training compute spent. Proceeding to attempt 3: one training
run, one variable off v1's exact recipe (add a revised, non-label-conflicting `_darkcontent`
dose).

### MISSION PLAN v3, Attempt 3 (2026-08-05 15:10:52 EEST): PASSES the gate — best mission result, ADOPTED as Phase 1's outcome

One variable off Attempt 1's exact recipe (same 250 orig + 250 `_blackbg`, seed 20260805, +50
`_darkcontent` pages with a revised floored transform avoiding Attempt 2's label-conflict bug).
Manual-clean 3-chapter result: aggregate 14.23% (over 3.37%/under 10.86%) — beats Attempt 1's
14.69% on both axes simultaneously, passes the plan's pre-committed gate. Honest caveat: the
specific HUD-panel flaw that motivated the attempt is essentially unchanged (28.16% vs. 27.66%
wrong-deletion in that exact region) — the improvement comes from a general softening elsewhere,
confirmed via direct visual comparison, not from fixing the targeted case. ROI battery: same
14/16 pass as Attempt 1, identical failure set, confirmed pre-existing (not newly introduced) via
a fresh baseline/v1/v3 three-way comparison. `blackbg_v3.pt` ADOPTED as Phase 1's outcome and the
new working base for Phase 2 (hole-detection hardening), closing the 3-attempt training-data
ladder with a genuine, if imperfect, net win — full progression: 20.19% baseline → 14.69%
(Attempt 1) → 25.39% regression (Attempt 2) → **14.23% (Attempt 3, adopted)**.

### MISSION PLAN v4, Step 1 (2026-08-05 15:45:57 EEST): HUD-panel defect is a RELEASE BLOCKER, ~18% of over-deletion, recurring class not an isolated page

Fresh visual review escalated the HUD-panel over-deletion caveat to a named release blocker.
Measured materiality by decomposing v3+islands' over-deletion on all 3 chapters and visually
classifying components: confirmed recurring instances beyond the original page (a night
cityscape, a dark cave scene with an embedded bubble, other dark dramatic scenes) — roughly 18%
of all over-deletion across the 3 chapters traces to this class, though under-deletion
(10.86pp) remains the dominant residual. Attempted to wire an automatic ROI check into
`ch002_rois.json` but found `data/chapters-initial/001.png` does not contain this page at all
(searched via component scan + naive scaling, neither matched) — the file is not a uniform
rescale of `.tmp/saved/chapters/001.png` despite matching width. Built
`.tmp/diagnostics/hud_panel_check.py` as a standalone tracked check instead (verified to
reproduce the known 28.16% result). Three-item release-blocker list now tracked in
`notes/next_session_handoff.md` through Phase 4. Proceeding to Phase 2a with Recipe A's effect
on this blocker as a mandatory Phase 2b line item.

### MISSION PLAN v5, Phase 2a (2026-08-05 16:20:24 EEST): hole-detector "1/5" was crop-window artifact — resolved with zero code changes

Per the plan's Step 0 (re-diagnose at full-page scale before any code change): built
`.tmp/diagnostics/hole_detector_stage_attribution.py` to attribute exactly which stage
(enclosure vs. classifier filter) loses each of the 5 tracked real instances. Result: 4/5 PASS
at full-page scale (the earlier "1/5" came from a 600×600 crop, confirmed as crop-window
artifact). Only inst2 fails, correctly rejected at the filter stage as genuinely frame-like
geometry (solidity 0.980, `is_frame=True`) — not an enclosure failure. Zero false holes on
skin/steam ROIs; 2 accepted holes found within the HUD-panel canonical region (blocker #1),
plausibly its own UI text boxes — promising but not yet quantified. Phase 2a CLOSED per the
plan's pre-committed ≥4/5 branch, zero code changes. Proceeding to Phase 2b.

### MISSION PLAN v5, Phase 2b blocker #1 (2026-08-05 16:38:04 EEST): repair_frame_interiors confirmed 0% no-op; offline solidity-mechanism simulation fails cleanly; ESCALATED per the plan's own trigger

`repair_frame_interiors` measured 0% effect on the HUD-panel canonical region across every Recipe
A chain step — direct trace confirmed it finds the right holes (UI text boxes, areas up to
179,669px) but they're already correctly kept by the model; the actual wrongly-deleted 28% is the
dark BACKGROUND itself, a defect shape `repair_frame_interiors` was never designed to address (it
protects light content enclosed by dark strokes, not dark strokes/fill being wrongly deleted).
Per the plan, activated the documented candidate (deletion-solidity + kept-adjacency on the
output mask), prototyped offline (cached masks, no inference) with an 18-combination sweep — FAIL
across all of them (HUD region flat at 28.13%). Root cause: the HUD panel's main dark component
has deleted_frac=0.446 (squarely in every tested band) but kept_adjacency=0.124, well below every
tested threshold — the hypothesized "patchy deletion clustered around kept content" signature
doesn't hold; the real pattern is diffuse deletion across a large area with sparse kept islands.
Escalated to Fable per the plan's own pre-declared trigger rather than spending attempt-budget
guessing at more variants — zero training/attempt budget consumed reaching this conclusion.

### MISSION PLAN v6 (2026-08-05 17:01:42 – 17:09:11 EEST): full-breadth blocker #1 campaign — class metric worse than the single case; D1 validates but doesn't touch the target class; D2 dead

3-attempt ceiling lifted for blocker #1 specifically (user authorization, contingent on
evidence-justified iterations). Guard step: `.tmp/diagnostics/darkpanel_class_check.py` measures
wrong-deletion across all 5 confirmed dark-panel-class regions, not just the canonical HUD page —
result 28.13% (canonical) / 46.30% / 36.65% / 34.29% / 36.51%, **class mean 36.37%**, substantially
worse than the single canonical case. New pass bar: canonical <5% AND class mean <8% AND
per-region visual pass.

Probe 0 (`.tmp/diagnostics/component_feature_table.py`): tabulated `deleted_frac`, `gt_precision`,
kept-island stats, and a new `soft_boundary_frac` feature (outer-ring intermediate-gray fraction,
testing whether dark art bleeds via soft glow/gradient vs. true gutters meeting hard frame/margin
edges) for all 100 near-black candidate components across 3 chapters, fit on ch001+002 only,
validated untouched on ch035 (same protocol that would have caught the earlier chroma
discriminator's overfit). **D1 (pure `deleted_frac` band-vote)** looked strong in isolation
(FIT margin 0.530, validates 89%/92% on held-out ch035) but simulating the actual rule
end-to-end (`.tmp/diagnostics/d1_bandvote_simulate.py`) shows it barely moves the target class
(canonical 28.13%→27.81%, class mean 36.37%→36.31%) — the class-check regions live inside
"mixed" components (gt_precision 0.2-0.8, deleted_frac 0.35-0.52) where true gutters and dark-art
panels are physically fused into one connected near-black blob, not the clean near-zero-deletion
dark_art population D1 actually discriminates well on. D1 does NOT close blocker #1, but is a
real, independently-verified ~1.0pp aggregate win (14.23%→13.20% total; over-del +0.37pp,
under-del -1.40pp) worth carrying forward as a candidate to combine with whatever eventually
resolves the class. **D2 (`soft_boundary_frac`)** failed outright — FIT margin -0.148, no
separation. Both dead ends feed D3: the "mixed"-component finding (gutter+dark-art fused by
connectivity) becomes design input for Family B's `_darkpanel`-coexisting-with-`_blackbg`
training construction, since a post-hoc component-level vote structurally cannot separate them.
Proceeding to Probe 1 (context-dependence tile-size sweep) next, per the plan's sequencing.

Probe 1 (`.tmp/diagnostics/probe1_tilesize_sweep.py`, 17:12-17:17 EEST): reran inference on all 5
class regions at tile_size ∈ {512,768,1024,1536,2048} with generous ±3500px context margins.
Class-mean swing across sizes = **0.66pp** (36.98%→36.34%→36.38%→36.32%→36.37%), max per-region
swing 1.28pp — cleanly CONTEXT-INDEPENDENT (plan's threshold: ≤2pp independent, ≥5pp dependent).
No inference-side lever exists; larger tiles/more context do not help. Confirms the plan's
featureless-middle hypothesis. Rules out inference knobs entirely — proceeding directly to Family
B (`_darkpanel` training-data fix), the only remaining mechanism per the plan's structure.

Family B variant built (`make_darkpanel_variant` in `make_black_backdrop_variant.py`, 17:20-17:24
EEST): authored dark-panel band (near-black, sparse bright UI boxes with dashed text-noise,
whole-band GT-keep) applied on top of an already-`_blackbg`-transformed page so real gutters and
the new panel coexist. Pre-training visual check (mandatory since this authors new GT) on 5
samples: PASS — bands show zero delete-overlay, surrounding real gutters still correctly delete,
no seam artifacts. Cleared for Rung 1 training.

**Rung 1 RESULT (`blackbg_v4.pt`, trained 17:29-18:23, gated 18:23-18:42 EEST): real regression,
both axes, ESCALATED.** Class mean 36.37%→**38.04%** (4 of 5 regions worse; canonical
28.13%→30.87%, dialogue_lightbeam 46.30%→48.98%, night_cityscape 36.65%→42.01%, dark_scene_text
36.51%→36.90%; only dark_cave_bubble improved, 34.29%→31.43%). Aggregate manual-clean
14.23%→**16.08%** (over-del 3.37%→8.06%, +4.69pp; under-del 10.86%→8.02%, -2.84pp) — over-deletion
more than doubling swamps the under-deletion gain. Both blow past the plan's Rung 1 gates. Direction
is the OPPOSITE of what the construction's rationale predicted (more over-deletion, not less) —
contradicts a written plan assumption, the plan's own escalation trigger. `blackbg_v4.pt` NOT
adopted; v3 remains the working base. Escalated to the user/Fable with 3 options (redesigned Rung
2 separating the dark-keep/dark-delete constructs onto different pages; a smaller dose to check
dose-linearity; or folding this into honest-exhaustion packaging) rather than Sonnet unilaterally
redesigning Family B. Full tables: `notes/instance_aware_pivot_2026-08-04.md`, 18:42 EEST section.

### MISSION PLAN v7 (2026-08-05 19:42-19:55 EEST): Phase 0 Bottleneck Content Probe — directional signal for CDR

NotebookLM literature round (critically filtered: SupCon/contrastive rejected per hardware/
feature-collapse risk; "flat fill" causal claim corrected against our actual noise+gradient
recipe, but the underlying "missing structural differentiator between authored panel band and
real gutter band" diagnosis independently verified). Plan v7: Phase 0 (bottleneck probe) → Phase
1 (Rung 2, frame-line data fix) → Phase 2 (CDR auxiliary reconstruction head, contingent).

Phase 0 (`bottleneck_probe.py`): froze `blackbg_v3.pt`'s encoder, trained a tiny decoder ALONE
on the bottleneck (synthetic data only, real chapters eval-only per discipline) to reconstruct
coarse grayscale structure, then evaluated zero-shot on 5 dark-panel-class crops vs 5 real
control crops. Result: both noisy vs. synthetic sanity (real domain-gap confound, expected), but
dark-panel MSE (mean 10862) consistently ~50% worse than control MSE (mean 7249) — 4/5 clean
separation, visually corroborated (controls retain shape echoes, dark-panel rows show near-pure
noise). Directional signal supporting CDR, not a clean binary — treated as decisive (relative
comparison, not absolute fidelity, was always the operative test). Phase 2 now evidence-motivated,
queued behind the cheaper, independent Phase 1.

### Plan v7 Phase 1 RESULT (2026-08-05 20:02-21:02 EEST): Rung 2 (`blackbg_v5.pt`, frame-line) — hypothesis CONTRADICTED, ESCALATED

Added a thin bright border line at the authored panel band's true edges (`frame_line=True`,
verified at full pixel resolution: uniform value 202 across full width, exactly at the boundary).
One variable vs Rung 1. Clean training (val_loss 0.287, best/final coincide). Result: class mean
WORSE (36.37%→38.04%→**39.00%**, 3/5 regions worse than Rung 1 too), aggregate flat vs Rung 1
(16.08%→**15.94%**, noise-level, still +1.71pp worse than v3's 14.23%). The missing-structural-
differentiator hypothesis is CONTRADICTED, not confirmed. Per the plan's own pre-declared trigger,
STOPPED and escalated rather than spending a 3rd run on the separate-pages rung unilaterally.
`blackbg_v5.pt` NOT adopted; v3 remains the working base. Phase 0's bottleneck-probe signal stands
independently — Phase 2 (CDR) remains live regardless of Family B's outcome.

### MISSION PLAN v8 (2026-08-06 20:56-21:13 EEST): user-directed classical-CV boundary completion — Phase 0 signal probe PASSES 6/6

Supersedes v7's open questions. User-provided real example (`.tmp/blackbg-border/`, ch035): a
dark panel border with asymmetric contrast (bright side obvious, dark-smoke side faint-but-real,
sharp not gradual cutoff) — hypothesis: real panel borders carry PARTIAL signal, not zero
everywhere. New mechanism class: classical CV boundary tracing + geometric completion, no model,
no training — sidesteps both Family B failure modes. Fully offline-developable against the
existing cached v3+islands masks.

Built `.tmp/diagnostics/border_signal_probe.py`: measures local Sobel gradient at each point along
the true GT panel-boundary contour, against a per-chapter measured noise floor (p99 gradient in
deep-interior GT-delete regions: 001=4.48, 002=26.91, 035=105.00), plus blind Hough-line
detectability (no GT) as a derisking check, across 6 instances (the 5 `darkpanel_class_check.py`
regions + the user's ch035 example, located via template match corr=0.9767).

**Caught and fixed a real methodology bug before trusting the first run**: initial pass showed
6/6 "generalizing," but overlay images revealed the traced contour ran down the crop's left/right
edges for full height — an artifact, since these are full-bleed webtoon panels spanning the entire
720px page width (no left/right gutter exists; only top/bottom transitions are real). Fixed by
dropping contour points within 3px of the true page canvas edge and segmenting the contour so
run-length stats don't merge across the removed gap.

**Post-fix result, all 6 instances generalize** (pre-committed bar >=4/6): each shows both real
strong and real weak stretches (frac_strong 0.27-0.90) with high blind-detectability (0.77-0.92) —
signal recoverable without GT. Visual review of all 6 overlays confirms no remaining artifacts;
the user's own ch035 example independently reproduces their hand description almost exactly (weak
red on the dark-smoke side, strong green on the bright side of the same transition), derived here
from first principles, not fit to it. Proceeding to Phase 1 (build the completion algorithm) per
the plan's pre-committed rule.

### Plan v8 Phase 1 (2026-08-06 21:13-21:24 EEST): completion algorithm built and self-tested, but fires ZERO completions on real instances — ESCALATED

Built `src/research/panel_border_completion.py` (Hough segment detection -> collinear grouping ->
gap-filling -> panel/gutter classification -> bounded reclaim). All 4 required safety self-tests
pass in isolation (interior-gap-only completion, never extrapolating past outermost anchors).
Smoke-tested against 2 real tracked instances: zero completions fired, even after relaxing
clustering/span tolerances — the panel/gutter side-separation safety gate correctly refused to
act because detected candidate lines sit inside already-correctly-kept art, not adjacent to a
solid gutter.

Root cause, confirmed by directly rendering the actual wrong-deletion shape (not guessing): the
real defect is a **large, mostly-contiguous wrongly-deleted blob** (e.g. nearly the entire
~4900px dark background around the HUD panel), not a thin strip near an under-completed border
line. Phase 0's GT-guided signal probe correctly found the boundary signal exists; Phase 1's
blind (no-GT) Hough detection on complex real art isn't reliably recovering the SAME full
enclosing contour, and even where it partially does, the originally-scoped ACTION (a depth-capped
local band reclaim) is the wrong shape of fix for a large irregular interior — closer to
`repair_frame_interiors`'s full-interior flood-fill than a bounded band.

ESCALATED per the plan's own pre-declared trigger rather than unilaterally redesigning toward a
bigger boundary-loop-closure + flood-fill rewrite. Options presented: (a) full redesign toward
loop-closure + flood-fill, (b) narrower fusion — use completion only to bridge gaps in
`repair_frame_interiors`'s own near-black stroke-ring detection (its documented blind spot) rather
than detecting the whole boundary from scratch, (c) fold into honest exhaustion, fall back to
Phase 2 (CDR) or accept v3+D1 as the practical stopping point.

### MISSION PLAN v9 (2026-08-06 21:30-21:49 EEST): full mandate — 3 forks decided; per-pixel patchy-deletion reclaim ADOPTED (partial blocker-#1 mitigation); Family B and CDR closed

User granted full decision mandate. Decisions: **(1)** blocker #1 → the handoff's own
long-documented leading candidate ("deletes patchily around kept content") at PER-PIXEL
granularity (loop-closure rewrite rejected — blind Hough unreliable on busy art; literal
ring-gap fusion rejected on a verified mismatch — `repair_frame_interiors` protects LIGHT
enclosed interiors, and this defect is near-black, part of the stroke map itself).
**(2)** Family B CLOSED (2/2 regressions, hypothesis contradicted, zero positive evidence).
**(3)** CDR CLOSED unbuilt (mission pattern 11-for-11: every learned mechanism failed
synthetic→real; every adopted win is geometric postprocessing).

**Phase A probe** (`patchy_deletion_probe.py`, fit 001+002 / hold out 035): blocker #1 splits
into two measured sub-classes. TEXTURED dark art separates in (kept-density-200px, max-Sobel-15px)
space — held-out 035 capture 0.889 at 0.298pp correct-leak (within the 0.3pp under-del budget).
FLAT dark digital paint (canonical HUD, lightbeam) has local-max Sobel p75=0 — locally
PIXEL-IDENTICAL to gutter, unfixable by any local rule; this explains the earlier
context-independence and fused-component findings and is the documented irreducible residual.

**Phases B-C** (`src/research/reclaim_patchy_deletion.py` + end-to-end offline gate, real
numbers): class mean 36.37%→**20.72%** (dark_cave_bubble 34.29→7.18, dark_scene_text
36.51→2.05, night_cityscape 36.65→21.71; canonical only 28.13→26.47 — the flat residual);
aggregate over 3.37→2.92%, under +0.298pp (budget PASS), total 14.2294→**14.0763%**. Visual
review of 24 rendered changed components: good reclaims are real art recovery; bad reclaims are
one consistent shape (gradient fade-to-black bands hugging panels, joining the existing
under-del class), no speckle, no catastrophic keeps. **ADOPTED** as `islands_patchy` chain in
`eval_gen6_checkpoint.py`. Fresh ch001 inference reproduces the cached simulation exactly
(11.2096%, 4-decimal match). Blocker #1: PARTIALLY MITIGATED, flat-paint residual formally
carried as a documented limitation; no further blocker-#1 mechanisms this mission.

### Plan v9 Phases F-H (2026-08-06 21:59-22:12 EEST): RECIPE A adopted at 12.4955% (from 20.19% original baseline); geometric ladder closed on a two-sided-ambiguity finding; one-command `--recipe stage1-v9` wired

**Recipe A** (assembled by one-step-at-a-time offline ablation, then confirmed by fresh
inference with EXACT 4-decimal reproduction): `islands → reclaim_patchy_deletion →
d1_region_vote → repair_frame_interiors → close_bubble_halo` on `blackbg_v3.pt`. Aggregate
over 2.5841% / under 9.9114% / **total 12.4955%** (chapters: 001 8.33%, 002 10.29%, 035
17.36%); class mean 20.71%, canonical 26.43%. Ablation verdicts: P+D order beats D+P; R adds
−0.56pp; H neutral (kept, established visual-defect step, zero cost); **SFX instance protect
EXCLUDED** (+1.50pp regression — its keeps are largely material the manual reference deletes).
Timing ~3-4 min/chapter full chain (soft target 8min: pass). ROI battery: v3's standing 14/16
(chain doesn't touch the checkpoint). Wired into `process_command` as `--patchy-reclaim`,
`--d1-vote`, and the one-command preset `--recipe stage1-v9`.

**Phase G residual decomposition + the closing finding**: residual is under-deletion-dominated;
ch035's under-del is 76.9% near-black full-width uniform strips — superficially
`reclaim_black_backdrop`'s exact target. One evidence-justified rung (adding B to the chain):
under-del −5.10pp but class mean 20.71→74.30%, total 12.50→13.91 — REJECTED, and the reason is
the mission's closing insight: **B's delete-target and blocker #1's keep-residual are the same
local pixel class (uniform near-black), split only by page-layout semantics** (035's strips are
GT-delete backdrop; 001's HUD scene is GT-keep art). The under- and over-deletion residuals are
two sides of one ambiguity no local/geometric mechanism can split — every further geometric move
buys one side at symmetric cost to the other. Geometric ladder CLOSED; the ~12.5% floor is the
measured limit of this checkpoint + local postprocessing under the project's constraints
(from-scratch, synthetic-only training, local compute). Final mission-wide progression:
20.19% (18.0-frames+islands) → 14.23% (blackbg_v3+islands) → **12.50% (Recipe A)**; the ≤5.0%
mission bar remains unmet and is documented as out of reach for this mechanism class — closing
it would require semantic page-layout understanding (panel-vs-gutter reasoning above the pixel
level), a different mechanism class than anything tried or available under current constraints.

### MISSION PLAN v10 (2026-08-06 22:26-22:47 EEST): region-level semantics probed — signal exists but is one-sided and already harvested; floor confirmed at region level; nothing adopted

User-directed test of the semantic-floor hypothesis: measure whether REGION/LAYOUT-level
features separate the near-black ambiguity before building anything.
`.tmp/diagnostics/region_semantics_probe.py` harvested all candidate components (d1 criteria,
GT-free) across 3 chapters, GT-labeled ART/BACKDROP/MIXED, 13 features each with the two
closed discriminators as controls, fit-001+002/holdout-035 split, both classes on both splits.
Results: (1) **band_density** (±300-row layout context) genuinely separates (non-overlapping
IQRs both splits) — region-level signal EXISTS, the first new-signal finding since the patchy
split; (2) **the delete side is locked at region level too** — the canonical HUD and
night-cityscape components sit inside the BACKDROP feature distribution on every measured
feature, so the ~5pp under-del prize is indistinguishability-locked at BOTH pixel (v9) and
region (v10) granularity; (3) the tracked class regions live in MIXED fused components,
unreachable by component votes. Keep-side-only vote built
(`src/research/semantic_region_vote.py`, band_density ≥0.75) and simulated end-to-end against
Recipe A: 21,210 px reclaimed at 100% precision but **−0.0074pp total** — the keep-side mass is
already harvested by patchy+D1. Sensitivity line at 0.60 reproduces the predicted false-ART
cliff (+1.01M bad px, total 12.82%). **NOT adopted** (noise-level win fails the cost/benefit
bar). Durable outcome: the ~12.5% floor now stands at both pixel and region level — the
strongest closure evidence in the project. Recipe A unchanged at 12.4955%.

### MISSION PLAN v11 (2026-08-06 22:47-22:56 EEST): slab-level (sub-component) smoke test — CLOSED as an honest negative; four-granularity indistinguishability, the floor is final

User-authorized full R&D into sub-component decomposition + scene membership, cheap-first.
Sliced all candidate components into horizontal slabs at dark-row runs (194 slabs, 3 chapters,
fit/holdout split), measured gap-height / flanking-content / gray / texture / model-deletion
per slab against GT labels. Findings: (1) aggregate slab separation exists but is redundant —
carried by material Recipe A already fixes; (2) the canonical comparison fails: the HUD's big
flat wrongly-deleted slabs (gap 522-626, gray 0, tex 0.03) are feature-identical to deep
backdrop slabs on every measured feature; (3) the scene-membership/flanking hypothesis is
measured dead (flanking content identical for both classes — in a vertical webtoon everything
is flanked); (4) the tall-gap sliver died on the mass check: a real 1548-row KEEP slab
violates the safety bar AND the tall-slab prize is only 0.17pp. Attempt B (learned) correctly
NOT spent — absent input signal, not expressiveness; scaling capacity on a signal problem is
the closed mistake. **The flat near-black ambiguity is measured indistinguishable at pixel
(v9), region (v10), component (v10), and slab (v11) granularity. The ~12.5% Recipe A floor is
final for this project's constraint envelope (from-scratch, MIT-clean, synthetic-only
training, local compute, no true scene understanding).**

### PLAN v12 (2026-08-07 11:35-11:56 EEST): pipeline-v2 manual procedure replicated as classical CV — ~94% deterministic, judgment share measured at 2.82% of page

User provided a complete manual-clean reference for `005-1.png` (source + cleaned RGBA + PSD +
prose procedure, `.tmp/scripts-manual/`). Task: determine whether the manual procedure is a
deterministic algorithm. PSD read as ground truth (img-layer mask == cleaned alpha, agreement
1.000000). Result: **standalone classical-CV script reproduces the manual reference at over
2.32% / under 0.49% / total 2.8159% of page.** Key findings: (1) the JSX 4-mask builder is
fully deterministic — calibrated pixel-semantics against the PSD's own mask rasters (Photopea
`adjustLevels` applies gamma as a DIRECT exponent; lightness desaturate; strict-> threshold;
square-kernel min/max; 99.2-99.7% agreement, residual = merge-time AA rendering); (2) the
hard-white magic-wand selection is perfectly rule-expressible (edge-touching + paper-white-in-
source components; 52 px seed error); (3) the soft-white extension only matches the reference
as per-pixel paper-white gating within the wand extent (wholesale component selection
over-deletes 5.9-7.9% in every fitted rule variant — the components genuinely merge background
with panel interiors, the prose's own "flood fill leakage"); (4) `repair_frame_interiors`
deterministically replaces half the manual frame-protection work (2.67pp); the rest — broken-
frame cases, crowned by a 373k-px framed white document panel whose ring is broken by an
overlapping bubble — plus SFX-outline/cloud-text brushwork is the measured judgment share.
Fitted-to-this-page thresholds; generalization untested and not claimed; white-bg domain only
(does not reopen v9-v11's closed dark-page ambiguity). Full addendum:
`.tmp/scripts-manual/pipeline-v2.md`; deliverable `replicate_pipeline_v2.py`.

### PLAN v13 (2026-08-07 12:24-12:32 EEST): v12 generalization measured DEGRADED (causes decomposed); flood-leak defect fixed (Canny barrier + validated leak post-filter, zero detected leakage on all 7 gold parts)

New gold set: 10 per-part PSDs (`.tmp/saved/psd/new-gold/`, May-era pipeline-v1 vintage, all
with clean img-layer masks >=99.96% binary). Dark check (mandated): 033-2/3/4 majority-dark
(dark frac 0.57/0.59/0.66) -> excluded from the white track; 033-1 NOT majority-dark (median
216, dark 0.248 -- same profile as 002-1) -> included FLAGGED. Part heights confirm gold-033 ==
harness "035".

**Phase 1 generalization (frozen 005-1-fitted constants, pre-stated bars all<=6%/median<=4%):
DEGRADED** — totals 5.0-34.7%/part, but decomposition shows the failure is under-deletion, in
two measured causes: (a) dark background (out of white-track scope entirely; up to 24.8pp on
001-1); (b) a REAL transfer failure: on these chapters bright mid-gray content (100-230) merges
with white gutters inside hard-white components (the hard-white recipe passes everything
brighter than ~gray 20), diluting frac250 to 0.45-0.80 -> seed rule misses -> 2.5-15.9%/part of
pure-white background never deleted (the background itself is just as pure as 005-1's --
white-point hypothesis measured and rejected). Over-deletion stayed 0.17-2.27% everywhere.
Note: the gold pages were manually cleaned with the v1 mask settings, not v2's.

**Phase 2 reference-free leak detector** (`leak_detector.py`, barrier-split: deleted
sub-components behind a closed near-black barrier with no page-edge connection): validated
against GT BEFORE use — **precision 1.000 on every part with detections** (~280k flagged px,
zero false positives); recall vs total over-deletion 0-24% by design (only border-crossing
leaks are in its domain; source-broken borders undetectable by construction — stated
limitation).

**Phase 3 leak fix (closed in exactly 3 attempts)**: (1) near-black-stroke barrier = measured
NO-OP by construction — gray<=40 px are already below the soft-white threshold (~43); floods
cross through BRIGHT bridges (steam/AA gaps), not dark pixels. (2) **Canny(60,120) barrier**
subtracted from soft-white before connectivity: −52% detected leakage overall, 5/7 parts to
zero, over-del down on EVERY part, under-del cost <=0.06pp/part. (3) **leak-detector
post-filter** (justified by its validated 1.000 precision): removes the remaining 135k px
(001-3's steam/soft-gradient cases) at zero under-del cost. After both: **zero detected
border-crossing leakage on all 7 parts**; canonical labeled example: bright-skin/steam flood
into a character's torso through a steam-broken outline (`.tmp/scripts-manual/leak_examples/`).
Deliverable: `replicate_pipeline_v3.py` (v2 left intact).

**Phase 4 comparative note — why SmallUNet leaks less at borders**: its 7-channel input
hard-codes edge awareness (threshold/morph-open/morph-close/CANNY guidance channels,
docs/decisions.md — deliberately encoding this same manual workflow), so every per-pixel
decision sees the edge map; the classical chain's flood is pure connectivity with no barrier
term until v3 added exactly that (a Canny barrier) — converging on the same design insight
from the opposite direction. The v3 post-filter has no SmallUNet analogue (topology-level
reasoning; nearest relative is reclaim_landlocked_delete_islands' inverse).

### PLAN v14 (2026-08-07 12:40-12:52 EEST): seed-merge fix ADOPTED as v4 (gold white-track error halved); synthetic breadth test surfaced the white-panel-interior ambiguity — one classical probe failed decisively; micro-net fallback flagged for explicit go-ahead

**Phase B (seed-merge fix)**: root cause confirmed on 001-2 (Canny barrier splits gradient-
rimmed content off cleanly, but soft gradient-free steam keeps merged components at frac250
0.50-0.94, below the 0.981 bar). Variants measured end-to-end (7 gold parts + fit page):
1a (split only) REJECTED — wholesale seeding of split-off panel whites violates the over-del
guard; **1b ADOPTED as `replicate_pipeline_v4.py`** — barrier-split of the hard mask +
gray>=250-GATED seeding + frac250 relaxed to 0.90: gold white-track median 8.77%->3.22%
in-experiment, over-del DOWN on every gold part, leakage stays 0. Flagged cost: the fit page's
judgment-class document panel re-exposed (+1.96pp over there; page 2.65%, within its 3.0%
bar). 1c (Canny 30/90) byte-identical no-op. **2x2 factorial completed 2026-08-08 (the
one-variable rule — 1b changed two coupled knobs)**: (gated, 0.981) = gating alone cuts
over-del (001-3 2.08->0.64%, fit 2.18->1.99%) with zero recall change; (wholesale, 0.90) =
relaxation alone delivers nearly all the recall (001-2 under-white 6.59%, 033-1 0.70%) at
+1-2pp over-del per part — measured NOT catastrophic, correcting an overcautious prediction.
Attribution: relaxation carries recall, gating carries safety; 1b composes both and remains
the best quadrant.

**Phase A (breadth: 20 synthetic GT pages via the P&C stage1 generator + full rerun)** —
pre-stated bars ALL FAIL, with three cleanly separated causes: (i) gold median 4.08% vs
bar 4.0, max 10.06% (001-1's white-pockets-in-dark-layouts — distinct cause, unchanged by the
seed fix); (ii) fit page 3.03% vs bar 3.0 (the doc-panel cost); (iii) **5 of 20 synthetic
pages over-delete 15-20%: full-bleed panels with pure-white interiors — locally IDENTICAL to
gutters (white band between two full-width bars), the project's original v1-v2 flood-fill
lesson in classical form. NOT a v4 regression: v3 fails the same pages WORSE (17-44%).**
15/20 synthetic pages: 0.02-0.5% (near-perfect).

**New-class classical probe (band-height rhythm): decisive negative** — tall-outlier band
exclusion (K x median, swept 1.5/2/3) fixed none of the synthetic outliers AND broke real
pages by +31-39pp under-deletion (real gutters vs text-gap-dominated medians). The
cross-distribution fragility demonstrates layout-statistics heuristics won't hold this class.

**Round verdict**: v4 strictly dominates v3 on every measured page and ships; the pre-stated
generalization bars remain unmet with the residual now attributed to two named open classes
(white-panel-interior ambiguity; white-pockets-in-dark-layouts). Recommended next step for the
white-panel class — the v14 brief's scoped micro-net local classifier (synthetic-only
training) — flagged for explicit go-ahead, not launched.

### PLAN v15 (2026-08-07 12:57-13:20 EEST): Track 2 diagnosis overturned the "dark pockets" label and closed it classically (v5); Track 1 micro-net = honest negative (no viable operating point); v13's generalization bars now PASS

**Track 2 (run first — its overlap verdict gated Track 1's scope)**: 001-1's 8.33% under-white
re-diagnosed. NOT white-pockets-in-dark-layouts (context measured bright: ctx_dark 0.01-0.08)
and NOT the band-ambiguity class — stage attribution showed the loss at the SEED stage: the
missed regions (chapter-title transition zones) live inside 1.1-2.5M-px merged components at
frac250 0.36-0.44, fused through SOFT gradients Canny cannot cut. The residual seed-merge
mechanism at page scale. Classical fix, attempt 1, ADOPTED: add the low-frequency gradient
(|grad(Gaussian sigma 8)| > 1.0) to the hard-mask split barrier. 001-1 under-white
8.33->1.14%, and it also closed 001-2's residual (6.62->0.50%). Flagged: 001-1 over-del
+0.85pp exceeds the +0.3pp guard letter, accepted on the 8.5:1 exchange + visual confirmation
the new over-del is the KNOWN white-inside-panel class. Partial overlap with Track 1 flagged
(the residual over-del belongs to that class in principle).

**Track 1 (micro band classifier, BandNet 24,691 params, synthetic-only fresh-seed data;
attempt log CORRECTED 2026-08-07 to the one-variable-per-run standard — the original
"attempt 2" conflated data scale and loss balancing; both were re-run in isolation on the
same data/seeds)**: attempt 1 (250 pages, plain BCE) froze at the 9:1 prior. Isolated 2a
(loss only: +pos_weight at 250 pages) breaks the freeze instantly but OSCILLATES violently
(gutter-recall 0.000-0.965) — the oscillation traces to the balancing, not the data.
Isolated 2b (data only: 1k pages, plain BCE) escapes the prior SLOWLY without balancing,
reaching balanced 0.849 by epoch 15 — data alone was nearly sufficient; balancing bought
speed at the cost of stability (attribution corrected from the original "imbalance + data
starvation" framing). Attempt 3 (lr 3e-4 + decay + best-checkpoint; best-checkpoint being measurement
bookkeeping) trained STABLY to val balanced-acc 0.874 (interior 0.957 / gutter 0.792); its
lr/scheduler bundle was ISOLATED 2026-08-08 — lr 3e-4 alone is stable to balanced 0.870 (the
lr reduction carries the stability), while 1e-3+StepLR alone still crashes early (epoch 7
balanced 0.545) and settles only once the decay reaches the low lr: the scheduler is
redundant polish. **Honest negative
at integration — verified on every variant**: the hook needs gutter-recall >= 0.995
(protecting one misclassified gutter band costs its whole area), and the operating-point
sweep collapses there on ALL trained variants (attempt 3: interior 0.00-0.11; 2a:
0.000-0.125; 2b: 0.109 at 1.0). Battery with the hook ON: fixed 2/5 target pages but
collapsed 7 previously-passing synthetic pages to 15-33% under-deletion. Hook shipped OFF by
default in `replicate_pipeline_v5.py`; checkpoints kept in
`.tmp/checkpoints/band_classifier/` (richer context is the untested axis). Classifier cost
when enabled: 137 bands in 804 ms on a 50k-row part.

**v5 shipping config final battery (hook off, all real numbers)**: gold white-track
001-1 3.72% (was 10.06 at v4), 001-2 0.78%, 001-3 1.32%, 002-1 2.48%, 002-2 2.46%,
002-3 1.80%, 033-1 7.01% (flagged part, +0.21 within guard); fit page 3.24% (+0.22, within
guard); 15/20 synthetic pages 0.02-0.48%. **The six verdict gold parts now sit at median
~2.1% / max 3.72% — v13's original generalization bars (all <=6.0%, median <=4.0%) PASS on
the fair white-track metric for the first time.** Open classes after v15: white-panel-interior
ambiguity (5 synthetic pages at 15-20%, Track 1's class, stays open) and the judgment-class
document panel (fit page).

### PLAN v16 (2026-08-07 17:30-17:53 EEST): seven user-found defect classes -> v6; regression battery PASS; Cluster 2 reuses Cluster 3's mechanism; two documented residuals

User's manual diagnostic pass produced 6 example crops (`.tmp/errors-examples/111-666.png`),
all template-located in ch007/008 (corr 0.978-1.000). Cluster grouping verified with one
correction: the claimed "cluster 1 vs 2" example split was as-suspected, and Cluster 3's four
findings are indeed one mechanism (local pockets) in different geometric contexts — EXCEPT the
666-class (see residuals).

**Measured bands (before any fix, per the discipline)**: JPEG border residue [240,250)
(76.6% of border-adjacent undeleted px ≥240 — user's hypothesis confirmed); under-frame line
210-220 (2,693/3,910 px in the 333-region histogram) — two SEPARATE bands, two separate
mechanisms, no shared-knob tension in practice.

**v6 = v5 + three post-steps** (`replicate_pipeline_v6.py`, v5 untouched):
A. border-residue sweep ([240,250) within 3px of deletion, 3 iterations);
B. under-frame line ([200,230] directly below near-black strokes, near deleted bg);
C. LOCAL BACKGROUND RECLAIM — the v15 gated-seed idea generalized to pockets: bright (≥240)
   pockets, ring ≥85% ink+deleted(+page-edge), area <10k (repair_frame_interiors' interior
   convention as the 3(d) size guard), closed-frame interiors excluded, plus the attempt-2
   OUTER-ring guard (context beyond the enclosing ink ≥35% deleted — separates floating SFX
   glyphs from letter counters in kept text; the corrected adversarial test shows 0 suspicious
   px in kept bubbles both regions, 4.5k px of SFX pockets correctly reclaimed). Cluster 2's
   variable-position edge line (found: col 688, 58-62k rows, both chapters; left-edge runs on
   007) is handled by the SAME step via a thin-line exemption (width ≤4px bypasses the area
   cap when edge-touching): residual 3,181→68 px on the test strip. No parallel code built.

**Regression battery (pre-stated guards) PASS**: gold deltas −0.20 to +0.11pp (four parts
improved), fit page −0.15pp (3.10%), all 15 passing synthetic pages stay ≤3%. Shipping file
verified to reproduce battery numbers exactly. **Per-step attribution completed 2026-08-08**
(ladder v5→+C→+C+A→+C+A+B through the full battery): C −0.061..+0.034pp/part (fit −0.021);
A −0.172..+0.069pp/part (fit −0.133 — removes JPEG residue counted as under-white); B
≤0.01pp on battery pages (its evidence is its target examples). Every rung individually
passes all guards.

**Rejected on measurement**: cluster-3 attempt 3 (2px ink dilation in ring composition) —
−3% residual on the 666-class for 9.4k px of leak into kept text counters.

**Documented residuals**: (1) the 666-class — large barrier-fragmented pockets around
free-floating note text (region improves only 64.2k→62.2k bright-undeleted px; safe local
rules can't reach it); (2) the white-panel-interior ambiguity + doc panel (unchanged, deferred
BandNet class). **BandNet-orthogonality statement (required deliverable): this round is fully
ORTHOGONAL to the deferred gutter-vs-white-interior class** — every v16 mechanism operates on
small local pockets/bands with unambiguous GT; none touches the full-width white-band label
ambiguity, and the 5 failing synthetic pages' numbers are byte-identical before/after v6.
Resuming BandNet later inherits v6 unchanged.

### PLAN v19 (2026-08-08 08:11-08:56 EEST): 007 etalon round — E3 frame-strip fix shipped (v7); halo-sweep and fused-gap mechanisms closed as honest negatives after full 3-attempt ladders

New reference triplet (`.tmp/minmax/007_{init,cleaned,etalon}.png`, alpha-decoded; the crops
are sub-pixel-resampled vs the source page, so evaluation uses an embedded-crop harness —
init crop pasted into the page at the located offset y=26389, verified 0.9952 mask agreement
with the user's own cleaned crop). True defect map: ZERO over-deletion vs the etalon; 4,884
px under-deletion in three classes: the page-edge fused gap (1,933 px, v16 222-class), the
gray-191 frame strip (365 px — below v16 B's [200,230] floor AND on the background side of
the frame line, i.e. related-but-distinct, not a regression), and floating-text AA halos
(user's manual recipe: select-by-color threshold 23 = gray>=232, applied locally by hand).

**Shipped (v7 = v6 + E3)**: frame-strip rule — band [185,230], within 2px of a >=100px
horizontal dark run, geodesically reachable from deleted background through the band. Etalon
-334 px at zero over-del; page-wide flags visually verified as background AA; battery PASS
with every gold part slightly improved.

**Honest negatives (full ladders, page-wide safety guards decisive)**: D halo sweep — three
mechanizations of the manual recipe all fail page-wide (17k/301k/87k suspicious px on 007);
the human's implicit "this is a floating-text region" selection has no safe classical proxy.
F fused-gap pockets — effective variants bite margin-adjacent art (35k suspicious), the safe
variant captures nothing; the class stays open. Both echo the standing lesson: locality
judgment, not pixel statistics, is the missing ingredient.

### PLAN v20 (2026-08-08 11:44-13:35 EEST): region-selection round for D/F — candidate GENERATION solved classically (recall 2/2 both classes); the ACCEPT decision failed classical rules AND scoped classifiers; both classes to the manual/GUI track

The round tested the one thing v19 never tried: mechanizing the manual recipes' REGION
SELECTION step before the sweep, with a scoped classifier held in reserve for the
accept/reject decision only.

**Sub-effort 1 — candidate generation: SOLVED CLASSICALLY.** `region_candidates.py`
(`.tmp/scripts-manual/`). D: glyph-scale ink components (20-4,000 px) dilate-clustered into
blocks + (attempt 2) a large-floating-component path (4k-200k px single components — the
synth diagnostic showed SFX letters at 40k-120k px are invisible to the glyph path) +
(attempt 3) min-glyphs 3->2 (the 444-sfx instance is a 2-component cluster). Recall 2/2
(007 translator-note ring_del=0.50; 008 444-sfx ring_del=0.85), 768/673 candidates per
chapter. F: bright-pocket components (>=240, undeleted, unprotected) with ring features;
attempt 2 raised the area cap 60k->120k (the 222-edge pocket is 88,446 px). Recall 2/2,
231/121 candidates per chapter. **Answer to the brief's question: candidate generation is
NOT the ML-requiring bottleneck — no ML needed for proposals.**

**Sub-effort 2a — classical accept rules: FAIL (pre-stated bars: total harm <= 5,000 px,
benefit >= 100,000 px, fit=001/002 holdout=033-1).** Ground truth: every gold candidate
labeled by measured benefit (would-sweep px that are GT-delete) vs harm (GT-keep). Pools
are ~94% harmful (D: 41 good vs 630 harmful of 857; F: 44 vs 558 of 610); total harm pool
3.5-4.1M px vs 219-272k benefit. Best D rule (ring_del>=0.7, ink<=0.05, prot=0): 52,621
benefit / 120,667 harm. Best F rule: ZERO benefit before harm floods. Feature IQRs overlap
at the quartiles; scalars cannot carry the decision.

**Sub-effort 2b — scoped classifiers (CandidateNet, 50,177 params, 3x96x96 context crop
[gray/v7-delete/protected] + 7 scalars, BCE pos-weight, Adam 3e-4, best-ckpt on synth val;
100% synthetic training per mandate: sfx-stage pages for D, bubbles-stage for F, 2,000
train + 200 val pages each; checkpoints in `.tmp/checkpoints/candidate_classifier/`, never
committed).**

- **F: STRUCTURAL NEGATIVE after attempt 1** (synth val prec@rec50 = 1.000 — the synthetic
  boundary is perfectly learnable — yet on gold the ranking INVERTS: top-30 scored = 26
  harmful / 3 good; good candidates score median 0.003; thr 0.9 buys 18k benefit at 449k
  harm). The montage diagnosis (`.tmp/scripts-manual/v20_f_gold_diag.log`, scratchpad
  montage): the top-scored harmful regions are the manhwa's SYSTEM-UI WINDOWS — dark
  panels, white borders, white text, semantically KEEP — which are locally identical to the
  synthetic "deletable bright pocket" pattern. A label collision in the training
  distribution itself (same local appearance, opposite label) plus wrong-looking positives
  (real good pockets are AA fringes/textured gradients, not P&C's clean white boxes). No
  single synthetic-only change fixes both directions; attempts 2-3 declared infeasible with
  the mechanism stated rather than burned blind.
- **D: HONEST NEGATIVE after the full 3-attempt ladder.** Attempt 1 (plain synth):
  transfers directionally (gold good median score 0.268 vs harmful 0.02 — correct ordering,
  unlike F) but no operating point: thr 0.5 = 123.8k benefit / 114.7k harm. Dominant harm
  mode = the same system-UI windows. Attempt 2 (composite synthetic UI-window negatives —
  dark panel + white border + white text, GT=keep — onto synth pages at extraction; the
  missing negative class IS synthesizable): harm now clears the cap at a real threshold but
  captures only 2,541 benefit of the 100k bar (fit 2,171/2,454; holdout 370/1,099). Attempt
  3 (+ scale alignment, synth resized 910->690 width to match real glyph/AA scale):
  2,425/3,362 — no material change. Needed ~40x more captured benefit at fixed harm; no
  remaining single-variable mechanism plausibly yields it.

**Verdict (pre-committed discipline): NO SHIP.** v7 unchanged, no v8 file created. D
(floating-text halos) and F (edge fused gaps) move to the manual/GUI track. Gutter/BandNet:
stays closed — the shared region structure this round found (proposals easy, accept
semantic) reproduces BandNet's own failure shape, so nothing is reopened. The etalon prizes
forgone: ~2.2k px (D) + ~1.9k px (F) per affected page-instance. The through-line now
proven at FOUR granularities and TWO paradigms (per-pixel, band, region-rule,
region-classifier): the accept decision requires knowing what the artist intended — the
~12.5% semantic floor's region-scale expression. Adversarial guard + battery were not
reached (nothing shipped). Full logs: `.tmp/scripts-manual/v20_*.log`.

### PLAN v21 (2026-08-08 15:29-16:05 EEST): spiky-cloud manual pipeline decoded — ACTION deterministic and verified, SCOPE is the locality wall (GUI track); border residue classified as a coverage GAP and FIXED (v8 ships = v7 + A')

**Task 1 — spiky clouds.** New reference material `spiky-clauds/` (019-2 paired crop +
binary-alpha etalon + PSD + Russian process note) decoded and VERIFIED, v12-style:
the manual Magic Wand tol=200/contiguous-OFF step is exactly "min-channel >= 55" (100.00%
of etalon-deleted px satisfy it, p0.1 = 57); the MinMax1/interior-fill steps exist to seal
the spiky contour and wholesale-keep its interior. Crop located exactly in 019.png at
(26, 77363) (template score 1.0000, byte-identical). v7 baseline there: 0 over / 19,268
under. **The ACTION given a human region is fully deterministic**: `clean_spiky_region`
(replicate_pipeline_v8.py) achieves over 0 / under ~300 above the frame line; the only
etalon disagreement (2,396 px) is below the frame line, in the zone the note itself says
was manually restored. **The SCOPE is not classically derivable** — 3-attempt ladder, all
negatives (a1 unscoped flood: INVALID as measured — leaky large-kernel geodesic crossed
thin barriers; corrected: +19.8M/+15.5M over-del on ch001/002 vs manual chapter GT; a2
gap-sealed-enclosure annulus + exact flood: safe but recovers only 31% of the crop target,
and on chapters +430k/+385k over for 80k/186k under; a3 non-connective in-annulus: crop
solved (under 96) but 53k suspicious in one 35k slab — false-positive "enclosures" include
forest art and a character's face). Choosing WHERE tol-200 applies = recognizing "floating
spiky SFX cloud" = the same object-level semantics as v20's D/F. Spiky clouds move to the
manual/GUI track with the verified action as the tool. Reuse statement: the action reuses
`_protected_interiors` + adds a 1px-close sealing step; no new paradigm.

**Task 2 — border residue: classification (a) GENUINE GAP, with evidence; fixed.** New
full-chapter manual etalons (`.tmp/minmax/other/`, AA alpha binarized at 128; 035 etalon
verified dy=0, bottom-trimmed 480 rows). Version ladder v5->v6->v7 on 001/002 border zones
(within 10px of >=100px horizontal dark runs), under-deletion by gray band: NO band worsens
at any rung (not a regression); v16-A closed [240,250) (2,582->280 on 001) and E3 closed
[185,230] (10,076->1,618), but [250,256) (26,705 / 53,694 px on 001/002) and [230,240)
(~2-3k) were never covered by any mechanism — a coverage gap. **Fix ladder (one variable
per attempt):** A'1 ungated [230,256) sweep: net NEGATIVE (over +505k on ch001 — 10x
overshoot, eats kept white AA page-wide); A'2 frame-zone gate: still net negative
(+35.8k over / -22.8k under); **A'3 sandwich gate (band px must be <=3px from BOTH dark
ink and existing deletion): first net-positive on all three chapters** — ch001 over
+17,425 / under -22,380; ch002 +11,383/-18,392; ch035 HOLDOUT +3,353/-5,286. The added
"over" is 90% thin (<=4px) frame-hugging slivers, median 2 px, only 1 px of GT-AA overlap
— visually verified on ch001 and 007 renders as the residue class itself (the etalon's own
1-2px hand-cleaning inconsistency), zero text/bubble/art content.

**Battery (v8 = v7 + A'3):** gold parts ALL improve (deltas -0.0022 to -0.0172pp, no
regressions; median 2.3663% vs 4.0% bar); fit 005-1 3.0911% (-0.0061pp); synthetic
passing-15 all <=0.17%, failing-5 byte-identical to v7 (deltas +0.0000pp — pre-existing
documented residuals, not A' effects); adversarial guard flags on 007/008 (5,806/4,034 px)
visually verified as the sliver class (80% thin, max component 116 px). SHIPPED as v8
default; `clean_page(steps='')` reproduces v7 exactly.

### PLAN v22 (2026-08-08 16:14-16:45 EEST): new-gold PSD method extraction — the black-track decoded; dark backdrop = the entire remaining gap; a3 flatness rule ships OPT-IN (v9, steps='D'), full-auto honest negative

**Part 1 — method extraction (new-gold PSDs, 10 parts, 26 layers).** The `mask-hard`/
`mask-soft` layers present in every PSD are BLACK-track candidate masks, not white-track:
bright-stratum recall vs final GT ~0-2% everywhere, dark-stratum recall 67-100%. Decoded
parameters: mask-hard ~ (gray <= 32) dilate r=4 (IoU 89.6% on 033-3), mask-soft ~
(gray <= 64) dilate r=4 (IoU 92.7%); 033-3's `black hard`/`black soft` refinements reach
prec 88/86% vs final GT. Crucially, the user applied these candidates SELECTIVELY:
near-wholesale on backdrop parts (033-2 dark recall 100%), rejected on dark-art parts
(001-2 precision 1%) — the scope decision was always human.

**Part 2 — 10-part v8 baseline (033-2/3/4 first-ever numbers).** White-track excellent
everywhere (0.0237-6.7978%); the dark stratum is the entire remaining gap: under-dark
24.8% (001-1), 11.9/5.7% (002-1/2), 12.1% (033-1), 38.5/35.1/41.7% (033-2/3/4) of page.
**Classical scope ladder (3 attempts, fit 001-1+033-3, holdout the rest):** a1
margin/deleted-bg connectivity — backdrop recall ~complete but dark ART eaten (001-2
over +14.0pp, zero benefit); a2 + wraps-panels protected-adjacency — fails (bubbles float
in dark art too; backdrop and art fuse into single components: 033-3 over 18.4pp on FIT);
a3 + flatness gate (std over 21x21 <= 2.0) — breakthrough on backdrop parts: full-error
001-1 28.6->7.3%, 033-3 36.6->6.8%, 002-1 14.2->4.9%, 002-2 8.1->5.2%, 033-1 18.9->9.9%,
033-2 38.5->4.7%, 033-4 44.0->11.9%; but dark-art parts REGRESS (001-2 +6.0pp, 001-3
+1.4pp, 002-3 +0.4pp) and the residual over-del on backdrop parts (renders) is in-scene
flat-black art fill (caption-box scene fields, blacks around figures) — locally identical
to page backdrop, distinguishable only compositionally. The semantic floor at the dark
stratum, consistent with the gen-6 blocker-#1 record.

**Verdict: full-auto dark track = honest negative; a3 ships OPT-IN.**
`replicate_pipeline_v9.py`: default == v8 byte-identical (verified; battery deltas all
+0.0000pp); `steps='D'` = `step_h_dark_backdrop` (flat dark connected to margins/deleted
bg, protected interiors excluded) for parts a human marks as backdrop-bearing — the
`--reclaim-islands` opt-in precedent; one click replaces the manual black-track pass at
~95% recall. Region-scoped GUI use available via the same machinery as v21's
`clean_spiky_region`.

**Part 3 — battery expanded to all 10 gold parts** (`v22_battery.py`; 033-2/3/4 added
with 2026-08-08 v8 references 0.0237/1.5246/2.3373). Bars restated to encode the standing
adjudications: failing-5 = unchanged-vs-documented (not <=3%), guard = within the
v21-adjudicated sliver counts (5,806/4,034 on 007/008). All bars PASS on v9-default.
No ML was trained on any real data; the PSDs served as GT and method documentation only.

### PLAN v23 (2026-08-08 18:47-19:20 EEST): spiky-cloud SCOPE solved — two-signal cascade at precision 1.0 (13/13 TP, 0/105 FP incl. holdout); ships OPT-IN as v10 steps='S' (fit-page bar blocked default-on via a documented GT-era conflict)

**Reference labeling (evidence-verified).** Gap-sealed-enclosure candidates (v21
proposal) on chapters 001/002: 102 candidates. Initial action-benefit/harm labels gave 0
TPs — diagnosed (render) as two artifacts: the harm at genuine clouds is (a) the bbox
margin's panel-edge band and (b) etalon-kept soup; relabeling on the benefit signature
(TP >= 10k px, FP < 2k, 4 mid-range excluded) lands EXACTLY on the brief's ground truth:
13 TPs = 3 on 001 + 10 on 002, with an empty benefit gap 7.8k-12.5k.

**Per-signal separability (all three reported, as mandated):**
  A radial run count (elliptical rim annulus 1.02-1.30, 360 bins): TP 60-103 runs vs FP
    max 42 — STRONG, the discriminating signal (empty gap [43,59] -> threshold 50).
    Its sub-statistics cv_gap and rim fraction: no separation.
  B spectral periodicity (FFT peak ratio): NO separation (TP 0.02-0.05 vs FP 0.01-0.09)
    — honest negative; the rim is dense irregular alternation, not periodic.
  C interior glyph count: partial (TP p10 8.6 vs FP p50 6) — used as second wall only.
**Cascade A(runs>=50) AND C(glyphs>=5): 13/13 TP, 0/85 FP, 0 FN.** Holdout (019 slab,
21 enclosures incl. v21's forest/face FPs, never seen by the cascade): first run MISSED
the known cloud — C counted 0 glyphs because the 019 cloud's text is dark-gray, not
near-black; one-variable fix C@ink<100 (reference confusion unchanged: 13/13, 0/85) ->
holdout PASS: exactly 1 site accepted (the real cloud), all 20 FPs rejected, and the
auto path reproduces v21's manual-bbox action numbers to within 4 px (2,392/304 vs
2,396/304). Guard: 007/008 full pages -> 5 accepted sites, every one rendered and
verified a genuine spiky cloud, deletions confined to between-ray soup.

**Fit-page finding (real, fixed, and the honest ship limiter).** The 005-1 battery bar
FAILED (+0.45pp): renders showed the action's bbox margin blanks a ~60px strip of
UNPROTECTED panel art below/above a cloud (the same band the 019-2 etalon hid inside its
manually-restored frame zone). Fix: `clean_spiky_region_clipped` (v10) — deletion limited
to the flood from bbox center that cannot cross a >=100px horizontal dark run. The panel
band is eliminated (renders); the remaining fit delta (+0.36pp) is ENTIRELY between-ray
soup that 005-1's 3-generations-old GT kept but the current recipe (verified by the
dedicated 019-2 etalon, 100.00% tol-200 consistency) deletes — a GT-era conflict on the
defect class itself, not content loss. Per discipline, no bar restatement on disputed GT:
**S ships OPT-IN** (v10 default == v9 exactly; steps='S', combinable with 'D'). If the
005-1 fit GT is ever refreshed with the current recipe, default-on can be revisited
honestly. Battery with S on: synthetic PASS, gold PASS (net improvement, three parts
better), fit blocked as above.

### PLAN v24 (2026-08-09 07:24-07:55 EEST): frame-junction nibble fixed (metric v1 retracted as artifact, AA-nibble metric adopted); frame-interior damage audited, attributed, and restored — v11 ships (default = v10 + step Q; S = frame-guarded action)

**Metric honesty first.** The initially-designed frame-continuity metric (dark-support
columns) showed 16/35 "broken" runs — but the pipeline deletes ZERO gray<=100 px anywhere,
and a source-only control run showed the same breakage (mean 91.5%, 10/19 runs <99% with
NO deletion): metric v1 measured source-line thinness, RETRACTED. Metric v2 (AA-nibble:
deleted px with 100<gray<=230 within ±2px of a >=100px frame run) is the adopted standing
bar for spiky-action changes.

**Issue 1 — frame-junction damage: CONFIRMED under metric v2 and fixed.** v10's S action
added 1,586/759/57/231 nibble px on 002/007/008/019-slab beyond the base pipeline. Fix
(attempt 1, the A'-style band pattern): ±3px protection band around detected frame runs
excluded from the spiky deletion (`clean_spiky_region_frameguard`, v11). Result: guard
removes 97-100% of the S-added nibble (residual 42/5/0/0 px); base-pipeline nibble
(1,652 px on 002, E3/A'-band class, battery-adjudicated in their rounds) is pre-existing
and out of the spiky action's scope. Attempt 2 (orientation ownership) NOT NEEDED.

**Issue 2 — frame-interior damage: audited, attributed, restored.** 10-part audit:
25,208 px of over-deletion INSIDE `_protected_interiors` (per part 0-5,615 px; the spiky
action contributes ZERO — identical with/without S). Attribution: A' 11,212 px + earlier
steps 13,996 px; geometry: top instances are thin slivers hugging the interior side of
frame lines (dilation-based conditions crossing thin lines — protection existed, steps
bypassed it). GT-legitimate in-interior deletion across ALL parts: 3,122 px, a single
thin empty box on 002-2 whose restore is white-on-white. **Reuse verdict: the user's
hypothesis holds — enforcing the existing interior detection suffices.** Fix: step Q
(`delete &= ~_protected_interiors`) in v11's DEFAULT path; 8:1 favorable trade measured
before shipping.

**Battery (v11 default): OVERALL PASS — every gold part improves or holds** (deltas
-0.0028..-0.0163pp; 033-4 exactly 0 as audited; 002-2 still net-improves at -0.0031pp
despite the restored box); fit page -0.0144pp; synthetic byte-stable; guard unchanged.
Deferred per brief: sealed-interior misclassification (user to provide crops).

## Methodology lessons (apply these before starting a new experiment)
1. **One variable group per training run.** Every regression that was hard
   to attribute (v7, v9) involved bundling multiple simultaneous dataset
   changes. When adding N new things, isolate at least the ones with any
   plausible interaction risk into separate runs. When a remedy is
   inherently multi-knob (a coupled pair, a stabilization bundle), complete
   the factorial — or isolate post-hoc on the same data/seeds — before the
   record is considered closed (2026-08-08 completions: BandNet lr/decay,
   v14 seed-rule 2x2, v16 per-step battery ladder; measurement bookkeeping
   like best-checkpoint selection is exempt, it is not a training variable).
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
