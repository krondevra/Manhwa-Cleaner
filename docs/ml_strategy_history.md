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

**Recommendation: keep `10.0-baseline` as the production checkpoint.** Both `16.0` attempts are
tracked in git for reference only (the final, non-buggy checkpoint at `data/models/16.0.pt`/`.json`
— the jitter-bug version was not kept, its finding is fully captured in this writeup and the code
fix). The core scale-mismatch diagnosis (2481px training vs ~700px production, directly measured,
not assumed) remains plausible and distinct from anything tried before — this session's failure to
realize the benefit is attributable to two identified, specific implementation issues (the
zoom-out padding bug, and the un-rescaled patch-size confound), not necessarily to the underlying
hypothesis being wrong. Unlike `13.0`'s mechanism (a genuine flaw in the *idea* as implemented) or
`15.0`'s severe multi-mechanism convergence (suggesting a structural ceiling), this one is more
honestly characterized as **inconclusive-on-the-idea, conclusive-on-two-implementation-bugs** —
worth a properly isolated retry (fixed jitter, patch-size scaled to match) before ruling the scale-
mismatch hypothesis out entirely.

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
5. **Real manhwa reference images are for geometry/style inspection only,
   never training pixels.** Two genuinely useful findings this session (the
   sci-fi UI-box style, the "real black panels have no border line" insight)
   came from *looking* at real references without ever using their pixels
   in the dataset.
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
