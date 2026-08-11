# Generation 8 — reference architecture (classical classifier composition)

Status: merged to `main` 2026-08-11. This document describes what runs, in what
order, and what each classifier contributes to the final keep/delete decision in
the fully composed chapter pipeline (`classifiers.sfx.clean_chapter_full`).
Individual validation records live in `docs/decisions.md` (per-commit) and
`docs/ml_strategy_history.md` (condensed per-round).

## Production defaults (UNCHANGED by generation 8)

- ML track: SmallUNet `data/models/10.0-baseline.pt` + `--reclaim-islands`.
- Classical spiky track: `src/spiky/pipeline.py` `clean_page_v10` (production),
  `clean_page` v12-ABES (candidate). `pipeline.find_spiky_sites` remains the
  production spiky caller — it has NOT been re-pointed at the framework profile.
- Everything below is ADDITIVE, opt-in API on top of these.

## The classifiers

- `classifiers/background.py` — reachability primitives (enclosed / flood /
  protected_interiors v1+v2), moved byte-identically out of the spiky pipeline
  (8.1.2). Used by: the spiky pipeline itself, the bubble-pocket keep, the
  spiky site action's protected interiors.
- `classifiers/frame.py` — validated page-scale LINE INVENTORY (morphological
  long-run detection + occlusion bridging + measured stroke thickness). Its
  rect grouping remains unvalidated and unused. Used by: panel segmentation,
  sfx band derivation, regular_cloud's alignment signal, sfx_glyph's
  line-structure exclusion.
- `classifiers/panel_segmentation.py` (8.11.1) — whole-chapter typed
  segmentation: row-blankness band decomposition reconciled with the line
  inventory -> ordered `gutter / panel / partial / borderless` segments +
  gutter-midpoint processing units. Panels are whole by construction.
- `classifiers/detector_framework.py` (8.4.1) — Profile = candidate generator
  + AND-voted geometric Signals; `detect(page, profile)`.
- `profiles/spiky_cloud.py` — v23 cascade port; site lists equivalence-proven
  identical to `pipeline.find_spiky_sites` on full chapters 002/019.
- `profiles/regular_cloud.py` (8.5.1) — the existing bubble classifier
  (`style_analysis.extract_enclosed_holes`) wrapped with two framework
  signals (thickness-aware frame-line alignment; rectangle-scoped stroke
  thickness). Suite: A 67/92, B 12/12, C 1/20.
- `profiles/sfx_glyph.py` (8.7.1) — isolated thin stroke structures; 4
  AND-voted signals (elong / w_p90 / iso_ink / bconc) + capped-thickness
  line-coverage exclusion. Refs recall 20/22, 0 harmful extras, synth 0/20.
- `classifiers/sfx.py` — the composition home. `clean_sfx_region` (page/crop
  scale, the decoded manual-recipe automation, 6-PSD-validated),
  `clean_chapter` (panel-aware driver), `clean_chapter_full` (everything).

## clean_chapter_full — order and authority

```
1. clean_chapter(rgb)                          the panel-aware sfx path:
   - panel_segmentation: units + keep extents  (panels, partials, dense
     borderless art islands kept wholesale)
   - gutter treatment per unit: pass-1 white (G >= 33, decoded constant)
     proposes deletion, minus sfx_glyph stroke keeps (connectivity-rescued
     under pass-2 min(RGB) >= 50, dilated by the measured 2/4 px Expand)
     and pocket bubble keeps (enclosed pockets >= 3000 px, +4 px)
2. regular_cloud keeps (per processing unit)   profile-accepted bubble regions
   (+4 px halo) subtracted from the delete. Regions overlapping a spiky site
   are EXCLUDED — conflict rule: the spiky deletion outranks a cloud keep
   (clouds classify into the same 'thorn' family; overlap is expected,
   counted in stats, resolved deterministically).
3. spiky_cloud site deletions (LAST)           profile-detected v23 sites get
   the production-validated site action `pipeline.clean_spiky_region_clipped`
   with `background.protected_interiors`. The ONLY in-panel delete authority
   in the composition; overrides every keep. Carried by the 12-instance
   suite's validation, including its protected-interior semantics (text-
   bubble interiors inside sites stay kept, the thorn fringe is deleted).
```

Final mask = step-1 delete, minus step-2 keeps, plus step-3 site deletions.

## Guards (layered)

- Chapter-scale adversarial: zero deleted px inside panel/partial interiors
  OUTSIDE spiky sites (measured 0 on 002/004 with all classifiers active).
- The 8.10.1 reactive guards (zero-line, blank-evidence, band-inversion) are
  BYPASSED on the segmentation-driven path (superseded; also measured
  insufficient at unit scope) but active for standalone `clean_sfx_region`.
- Standing regression gates: full battery + 12-instance spiky suite +
  6-reference SFX suite (`classifiers/tests/sfx_suite.py`), all PASS.

## Validation boundary (honest scope)

Everything above is validated on ONE series' layout: vertical webtoon strip,
width 690, light/pink gutters, bordered-panel-dominant, plus the 6 manual-clean
reference PSDs from the same workflow. Structurally different layouts (grid/
tiled panels, dark gutters, borderless-dominant chapters) are UNTESTED; the
dark-background domain is PAUSED pending manual reference PSDs and would break
the row-blankness signal specifically. Known gaps, not silent assumptions.
