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
  inference-side-fixable artifact. Two levers tried and ruled out:
  `--boundary-patch-ratio` at 0.5 (model 10.0) did not resolve it and
  looked mildly worse on 2/3 test crops; increasing model capacity
  `base_channels` 24→64 (model 12.0) measurably worsened it on 2/3 test
  crops instead of helping. The white-bg-only recipe simplification
  (10.0-baseline) remains the only change that helped — worth
  investigating why before trying another sampling- or capacity-side fix
  (e.g. is it simply "fewer, more consistent variants → less competing
  signal", which would point toward dataset composition as the more
  promising lever over both sampling strategy and capacity).
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
