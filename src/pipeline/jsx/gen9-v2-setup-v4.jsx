// Photopea JSX — Manhwa-Cleaner gen9 v2 pipeline setup (steps 1-24)
// v4 (9.16.02, 2026-08-16): v3 and v2 died at the SAME instruction
//      (byte-identical partial PSDs) — the shared assertChanged() guard:
//      Photopea has no layer.histogram, so the guard itself threw an
//      uncaught TypeError with no alert. v4 makes the guard defensive
//      (skips when histogram is unavailable) and stage-tags layer names
//      ("name@levels", "name@thr", "name@slam") so a crashed PSD shows
//      exactly where it stopped.
// v3 (9.16.01, 2026-08-16). History:
//  v1: applyLevels() silently no-op'd (both call paths) and Photopea's
//      Threshold uses ~Rec.601 luma with AA -> 31%/13%/2% divergence.
//  v2: Levels as adjustment layer WORKED (proven by the partial PSD:
//      8 corner colors on the merged layer), but Photopea REJECTED the
//      Channel Mixer descriptor -> script died mid context-fill.
//  v3: mixer dropped. Only mechanisms PROVEN to execute in Photopea:
//      Levels-AdjL + merge (proven in v2), Threshold-AdjL + merge
//      (proven in v1), Minimum/Maximum (proven in v1).
//
// WHY THE RE-TUNED THRESHOLDS ARE EXACT (0 px vs the manual reference
// in simulation, under both plausible 601 variants):
//   After Levels with a 1-unit window, every pixel is one of 8 corner
//   colors (each channel 0 or 255). Photopea-Threshold's Rec.601 luma
//   of the corners: white 255, yellow 226, cyan 179, green 150,
//   magenta 105, red 76, blue 29, black 0. The reference (Rec.709 on
//   leveled RGB) classifies corner SUBSETS, and each subset boundary
//   falls in a wide 601 gap:
//     outlines: white <=> {white,yellow}          -> cut (179,226] -> 200
//     SFX:      white <=> {white,yellow,cyan,grn} -> cut (105,150] -> 128
//     context-fill: white <=> {white}             -> cut (226,255] -> 240
//   A final Levels(127,1,128) merge slams Photopea's threshold-edge AA
//   to strict {0,255} (may be a legit no-op if no AA was produced).
//
// End state, layer stack top -> bottom:
//   context-fill   (Levels 160,1,161 -> pThr 240 -> Minimum 3 -> Maximum 3)
//   outlines-SFX   (Levels 120,1,121 -> pThr 128)
//   outlines       (Levels 33,1,34   -> pThr 200)
//   img            (untouched original, receives the raster mask step 26+)
//   fill           (solid red, bottom, red-preview reference)

var doc = app.activeDocument;

//////////////////////////////////////////////////
// SETTINGS — from new-pipeline-classifier-v2
// photopeaThreshold is TUNED FOR PHOTOPEA'S 601 LUMA (see header);
// the algorithm-text thresholds (226/128/250) assume Rec.709 and are
// implemented by these equivalents exactly.
//////////////////////////////////////////////////

var contextFillSettings = {
    name: "context-fill",
    levelsBlack: 160,
    levelsWhite: 161,
    photopeaThreshold: 240,   // algorithm: 250 @ Rec.709
    minimumRadius: 3,
    maximumRadius: 3,
    morphOrder: "minmax" // step 15 Minimum -> step 16 Maximum
};

var outlinesSfxSettings = {
    name: "outlines-SFX",
    levelsBlack: 120,
    levelsWhite: 121,
    photopeaThreshold: 128,   // algorithm: 128 (already in the 601 gap)
    minimumRadius: 0,
    maximumRadius: 0,
    morphOrder: "none"
};

var outlinesSettings = {
    name: "outlines",
    levelsBlack: 33,
    levelsWhite: 34,
    photopeaThreshold: 200,   // algorithm: 226 @ Rec.709
    minimumRadius: 0,
    maximumRadius: 0,
    morphOrder: "none"
};

//////////////////////////////////////////////////
// HELPERS
//////////////////////////////////////////////////

function fail(step, e) {
    var msg = "gen9-v2-setup FAILED at: " + step +
        "\nactive layer: " + doc.activeLayer.name + "\n" + e;
    alert(msg);
    throw new Error(msg);
}

function cT(s) { return charIDToTypeID(s); }

function makeAdjustmentLayer(typeID, settingsDesc, stepName) {
    try {
        var desc = new ActionDescriptor();
        var ref = new ActionReference();
        ref.putClass(cT("AdjL"));
        desc.putReference(cT("null"), ref);
        var adjDesc = new ActionDescriptor();
        adjDesc.putObject(cT("Type"), typeID, settingsDesc);
        desc.putObject(cT("Usng"), cT("AdjL"), adjDesc);
        executeAction(cT("Mk  "), desc, DialogModes.NO);
    } catch (e) {
        fail(stepName, e);
    }
}

// Levels (inputBlack, gamma 1, inputWhite), composite channel, as an
// adjustment layer. PROVEN working in Photopea (v2 partial PSD).
function makeLevelsAdjustment(inputBlack, inputWhite, stepName) {
    var lvl = new ActionDescriptor();
    var adjs = new ActionList();
    var one = new ActionDescriptor();
    var chn = new ActionReference();
    chn.putEnumerated(cT("Chnl"), cT("Chnl"), cT("Cmps"));
    one.putReference(cT("Chnl"), chn);
    var inp = new ActionList();
    inp.putInteger(inputBlack);
    inp.putInteger(inputWhite);
    one.putList(cT("Inpt"), inp);
    one.putDouble(cT("Gmm "), 1.0);
    var otp = new ActionList();
    otp.putInteger(0);
    otp.putInteger(255);
    one.putList(cT("Otpt"), otp);
    adjs.putObject(cT("LvlA"), one);
    lvl.putList(cT("Adjs"), adjs);
    makeAdjustmentLayer(cT("Lvls"), lvl, stepName);
}

// Threshold adjustment layer. PROVEN working in Photopea (v1 output
// was demonstrably thresholded).
function makeThresholdAdjustment(value, stepName) {
    var thr = new ActionDescriptor();
    thr.putInteger(cT("Lvl "), value);
    makeAdjustmentLayer(cT("Thrs"), thr, stepName);
}

function mergeActiveLayerDown(stepName) {
    try {
        doc.activeLayer.merge();
        return;
    } catch (e1) {
        try {
            executeAction(cT("Mrg2"), undefined, DialogModes.NO);
        } catch (e2) {
            fail(stepName + " (merge)", e2);
        }
    }
}

function applyMinimumPX(radius) {
    if (radius <= 0) return;
    try {
        var desc = new ActionDescriptor();
        desc.putUnitDouble(cT("Rds "), cT("#Pxl"), radius);
        executeAction(cT("Mnm "), desc, DialogModes.NO);
    } catch (e) {
        fail("Minimum " + radius + "px", e);
    }
}

function applyMaximumPX(radius) {
    if (radius <= 0) return;
    try {
        var desc = new ActionDescriptor();
        desc.putUnitDouble(cT("Rds "), cT("#Pxl"), radius);
        executeAction(cT("Mxm "), desc, DialogModes.NO);
    } catch (e) {
        fail("Maximum " + radius + "px", e);
    }
}

function applyMorphology(settings) {
    if (settings.morphOrder === "none") return;
    if (settings.morphOrder === "minmax") {
        applyMinimumPX(settings.minimumRadius);
        applyMaximumPX(settings.maximumRadius);
        return;
    }
    if (settings.morphOrder === "maxmin") {
        applyMaximumPX(settings.maximumRadius);
        applyMinimumPX(settings.minimumRadius);
        return;
    }
    fail("morphology", "unknown morphOrder: " + settings.morphOrder);
}

function fillRed() {
    var redColor = new SolidColor();
    redColor.rgb.red = 255;
    redColor.rgb.green = 0;
    redColor.rgb.blue = 0;
    doc.selection.selectAll();
    doc.selection.fill(redColor);
    doc.selection.deselect();
}

// layer.histogram is NOT implemented in Photopea (the v2/v3 crash) —
// return null when unavailable and skip the guard.
function layerHistogram() {
    try {
        var h = doc.activeLayer.histogram;
        if (h && h.length === 256) return h;
    } catch (e) {}
    return null;
}

// Guard against silent no-ops (the v1 failure mode). Only used on steps
// that MUST change the layer; skipped when histograms are unavailable.
function assertChanged(stepName, beforeHist) {
    var h = layerHistogram();
    if (h === null || beforeHist === null) return; // guard unavailable
    var same = true;
    for (var i = 0; i < 256; i++) {
        if (h[i] !== beforeHist[i]) { same = false; break; }
    }
    if (same) fail(stepName, "histogram unchanged — stage was a no-op");
}

// Builds one calibrated mask layer from the untouched "img" source.
function buildMaskFromImg(sourceLayer, settings) {
    doc.activeLayer = sourceLayer;
    var maskLayer = sourceLayer.duplicate();
    maskLayer.name = settings.name;
    doc.activeLayer = maskLayer;

    // 1. Levels (lo, 1, hi) per channel -> merge (proven mechanism)
    var h0 = layerHistogram();
    makeLevelsAdjustment(settings.levelsBlack, settings.levelsWhite,
        settings.name + ": Levels(" + settings.levelsBlack + "," +
        settings.levelsWhite + ")");
    mergeActiveLayerDown(settings.name + ": Levels merge");
    assertChanged(settings.name + ": Levels", h0);
    doc.activeLayer.name = settings.name + "@levels";

    // 2. Photopea Threshold at the 601-tuned cut -> merge (proven)
    var h1 = layerHistogram();
    makeThresholdAdjustment(settings.photopeaThreshold,
        settings.name + ": Threshold(" + settings.photopeaThreshold + ")");
    mergeActiveLayerDown(settings.name + ": Threshold merge");
    assertChanged(settings.name + ": Threshold", h1);
    doc.activeLayer.name = settings.name + "@thr";

    // 3. AA slam: Levels(127,1,128) -> merge. Legit no-op if Photopea
    //    produced no AA, so no assertChanged here.
    makeLevelsAdjustment(127, 128, settings.name + ": AA slam");
    mergeActiveLayerDown(settings.name + ": AA slam merge");

    doc.activeLayer.name = settings.name;
    maskLayer = doc.activeLayer;

    applyMorphology(settings);

    doc.activeLayer = maskLayer;
    return maskLayer;
}

//////////////////////////////////////////////////
// MAIN — steps 1-24
//////////////////////////////////////////////////

// Step 1: rename source layer to "img"
var imgLayer = doc.activeLayer;
try { imgLayer.isBackgroundLayer = false; } catch (e) {}
imgLayer.name = "img";

// Steps 2-4: create "fill" layer (solid red), move to bottom
var fillLayer = doc.artLayers.add();
fillLayer.name = "fill";
doc.activeLayer = fillLayer;
fillRed();
try {
    fillLayer.move(imgLayer, ElementPlacement.PLACEAFTER);
} catch (e) {}

// Steps 5-24: the three calibrated masks, each duplicated from img
var contextFillLayer = buildMaskFromImg(imgLayer, contextFillSettings);
var outlinesSfxLayer = buildMaskFromImg(imgLayer, outlinesSfxSettings);
var outlinesLayer = buildMaskFromImg(imgLayer, outlinesSettings);

//////////////////////////////////////////////////
// Final layer order (top -> bottom):
// context-fill / outlines-SFX / outlines / img / fill
//////////////////////////////////////////////////

try { outlinesLayer.move(imgLayer, ElementPlacement.PLACEBEFORE); } catch (e) {}
try { outlinesSfxLayer.move(outlinesLayer, ElementPlacement.PLACEBEFORE); } catch (e) {}
try { contextFillLayer.move(outlinesSfxLayer, ElementPlacement.PLACEBEFORE); } catch (e) {}

//////////////////////////////////////////////////
// Visibility — hide the masks, show img + fill
//////////////////////////////////////////////////

contextFillLayer.visible = false;
outlinesSfxLayer.visible = false;
outlinesLayer.visible = false;
imgLayer.visible = true;
fillLayer.visible = true;
doc.activeLayer = imgLayer;

alert(
    "gen9 v2 setup complete (steps 1-24).\n\n" +
    "Layer stack ready:\n" +
    "  context-fill (160,1,161 / pThr 240 / min3-max3)\n" +
    "  outlines-SFX (120,1,121 / pThr 128)\n" +
    "  outlines (33,1,34 / pThr 200)\n" +
    "  img\n" +
    "  fill (red)\n\n" +
    "Continue manually from step 25:\n" +
    "Magic Wand on 'outlines' -> raster mask on 'img', etc."
);
