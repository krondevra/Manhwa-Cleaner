"""Gen9 v2 staged runner: the user's sequential hard-lock hierarchy
(frames -> SFX -> spiky cloud), ported stage-for-stage against the
006-crop checkpoint PSDs (decisions.md 2026-08-14).

    .venv/bin/python src/gen9/run_hierarchy.py <page.png> [out_dir]

Stages (checkpoint each maps to):
  S1  layers: outlines / context-fill / SFX          (deterministic)
  S2  base background delete (A2) + frame lock       -> before-26
  S3  field-speck compensation delete                -> before-30
  S4  trapped-pocket delete (C)                      -> before-32
  S5  SFX fringe restore (B', expand-4) + SFX lock   -> before-44
  S6  spiky halo re-classify (D zone)                -> before-49
  S7  spiky interior restore (expand-1)              -> before-53
  S8  interior hole fill restore + spiky lock        -> final etalon

PageState enforces the write-once locks; verify_locks() runs after every
stage. No decision-maker exists outside classifiers A2/B'/C/D and the
two wand-derived px rules (S3 specks, S6 halo) -- adding one is a
stop-and-report event.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

ISLAND_MAX = 120000   # kept comps up to this float in the bg zone
                      # (006 crop: islands <= 37k, panels >= 315k)
BAND_R = 16           # near-field writable band (GT speck max dist 12)


def run_page(page_png, out_dir=None, keep_top_band=True, src=None):
    import cv2

    from gen9 import classify_bg, classify_sfx, classify_spiky
    from gen9 import pipeline as pl
    from gen9.page_state import PageState

    if src is None:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        page_png = Path(page_png)
        src = np.array(Image.open(page_png).convert("RGB"))

    report = {}
    snaps = {}

    # S1 -- deterministic layers
    outl = pl.outlines(src)
    cf = pl.context_fill(src)
    sfxl = pl.sfx_layer(src)
    sfx_ink = sfxl < 128

    # S2 -- base background delete (Classifier A2) ------------------------
    n, lab, st = pl.bg_components(cf)
    sel = classify_bg.select_background(lab, st, keep_top_band=keep_top_band,
                                        src=src)
    state = PageState(src.shape[:2])
    state.delete(pl.compose_delete(src.shape[:2], lab, sel, outl))
    report['bg_comps'] = len(sel)

    # Classifier D candidates (needed for the frame-lock carve: the cloud
    # is an element in the background area, not frame territory)
    clouds = classify_spiky.find_spiky(cf, sfxl, state.mask, lab, st, sel)
    report['clouds'] = [dict(comp_id=c['comp_id'], crossings=c['crossings'],
                             rect=c['rect']) for c in clouds]

    # frame lock = kept structures minus the writable background zone
    kept = ~state.mask
    nk, klab, kst, _ = cv2.connectedComponentsWithStats(
        kept.astype(np.uint8), connectivity=8)
    islands = np.isin(klab, [i for i in range(1, nk)
                             if kst[i, 4] <= ISLAND_MAX])
    band = cv2.dilate(state.mask.astype(np.uint8),
                      np.ones((2 * BAND_R + 1,) * 2, np.uint8)).astype(bool)
    carve = islands | (band & kept)
    zones = []
    for c in clouds:
        z = classify_spiky.spiky_zone(c, lab, st, sel)
        zones.append(z)
        carve |= z
        state.add_pending(f"spiky-{c['comp_id']}", 'spiky',
                          rect=c['rect'])
    frame_lock = kept & ~carve
    state.lock_frames(frame_lock)
    snaps['S2'] = state.mask.copy()

    # S3 -- field-speck compensation --------------------------------------
    spiky_all = np.zeros(src.shape[:2], bool)
    for z in zones:
        spiky_all |= z
    specks = classify_sfx.find_field_specks(src, state.mask,
                                            exclude=spiky_all,
                                            sfx_ink=sfx_ink)
    state.delete(specks, tag='S3-specks')
    report['speck_px'] = int(specks.sum())
    state.verify_locks()
    snaps['S3'] = state.mask.copy()

    # S4 -- trapped pockets (Classifier C; B' selection adjudicates) ------
    slab, ssel, srows = classify_sfx.select_sfx_comps(sfxl, snaps['S2'], cf)
    for r in srows:
        state.add_pending(f"sfx-{r['comp_id']}", 'sfx', bbox=r['bbox'])
        if not r['selected']:
            state.resolve(f"sfx-{r['comp_id']}", 'keep')
    pockets, prows = classify_sfx.select_pockets(src, state.mask, slab, ssel)
    state.delete(pockets, tag='S4-pockets')
    report['sfx_selected'] = len(ssel)
    report['pockets'] = prows
    state.verify_locks()
    snaps['S4'] = state.mask.copy()

    # S5 -- SFX fringe restore + lock -------------------------------------
    selmask = np.isin(slab, ssel)
    fringe = classify_sfx.expand_fringe(selmask) & state.mask
    state.restore(fringe, tag='S5-fringe')
    for i in ssel:
        state.resolve(f'sfx-{i}', 'restore-fringe')
    state.lock_sfx(classify_sfx.expand_fringe(selmask) | pockets)
    report['fringe_px'] = int(fringe.sum())
    state.verify_locks()
    snaps['S5'] = state.mask.copy()

    # S6 -- spiky halo re-classify ----------------------------------------
    for z in zones:
        bg_px, fg_px = classify_spiky.halo_classify(src, z)
        state.delete(bg_px & ~state.mask, strict=False, tag='S6-halo-del')
        state.restore(fg_px & state.mask, strict=False, tag='S6-halo-res')
    state.verify_locks()
    snaps['S6'] = state.mask.copy()

    # S7 -- interior restore (expand-1) -----------------------------------
    for c in clouds:
        exp = cv2.dilate(c['interior'].astype(np.uint8),
                         classify_spiky.SQ3).astype(bool)
        state.restore(exp & state.mask, strict=False, tag='S7-interior')
    state.verify_locks()
    snaps['S7'] = state.mask.copy()

    # S8 -- interior hole fill + spiky lock -------------------------------
    for c, z in zip(clouds, zones):
        exp = cv2.dilate(c['interior'].astype(np.uint8),
                         classify_spiky.SQ3).astype(bool)
        holes = classify_spiky.fill_holes(exp) & ~exp
        state.restore(holes & state.mask, strict=False, tag='S8-holes')
        state.resolve(f"spiky-{c['comp_id']}", 'halo-deleted-interior-kept')
    state.lock_spiky(spiky_all)
    state.verify_locks()
    snaps['S8'] = state.mask.copy()
    report['clipped'] = state.clipped_log
    report['deleted_px'] = int(state.mask.sum())

    if out_dir is not None:
        from PIL import Image
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(page_png).stem if page_png is not None else 'page'
        delete = state.mask
        np.save(out / f'{stem}_gen9v2_delete.npy', delete)
        red = src.copy()
        red[delete] = (red[delete] * 0.45 + np.array([255, 0, 0]) * 0.55
                       ).astype(np.uint8)
        Image.fromarray(red).save(out / f'{stem}_gen9v2_red.png')
        clean = src.copy()
        clean[delete] = 255
        Image.fromarray(clean).save(out / f'{stem}_gen9v2_clean.png')

    return state, snaps, report


if __name__ == '__main__':
    _, s, rep = run_page(sys.argv[1],
                         sys.argv[2] if len(sys.argv) > 2 else '.tmp/gen9/out')
    print(rep)
