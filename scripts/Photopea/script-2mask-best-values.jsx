// Photopea / Photoshop JSX - grayscale-only stable version
// Start state after reload: Background
// Final expected state: mask-soft / mask-hard / img / red
//
// HARD uses best grayscale candidate from ROI search:
// Levels 26 / 0.40 / 110, Threshold 32, Minimum 14, Maximum 14
// SOFT uses best soft candidate:
// Levels 10 / 0.60 / 110, Threshold 64, Minimum 10, Maximum 10

var doc = app.activeDocument;

var hardSettings = {
    name: "mask-hard",
    levelsBlack: 26,
    levelsGamma: 0.40,
    levelsWhite: 110,
    threshold: 32,
    minimumRadius: 14,
    maximumRadius: 14
};

var softSettings = {
    name: "mask-soft",
    levelsBlack: 10,
    levelsGamma: 0.60,
    levelsWhite: 110,
    threshold: 64,
    minimumRadius: 10,
    maximumRadius: 10
};

function fillRed() {
    var redColor = new SolidColor();
    redColor.rgb.red = 255;
    redColor.rgb.green = 0;
    redColor.rgb.blue = 0;
    doc.selection.selectAll();
    doc.selection.fill(redColor);
    doc.selection.deselect();
}

function createThresholdAdjustment(value) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();
    ref.putClass(charIDToTypeID("AdjL"));
    desc.putReference(charIDToTypeID("null"), ref);
    var adjDesc = new ActionDescriptor();
    var thresholdDesc = new ActionDescriptor();
    thresholdDesc.putInteger(charIDToTypeID("Lvl "), value);
    adjDesc.putObject(charIDToTypeID("Type"), charIDToTypeID("Thrs"), thresholdDesc);
    desc.putObject(charIDToTypeID("Usng"), charIDToTypeID("AdjL"), adjDesc);
    executeAction(charIDToTypeID("Mk  "), desc, DialogModes.NO);
}

function applyLevels(inputBlack, gamma, inputWhite) {
    try {
        doc.activeLayer.adjustLevels(inputBlack, inputWhite, gamma, 0, 255);
        return;
    } catch (e0) {
        try {
            var desc = new ActionDescriptor();
            var list = new ActionList();
            var lvlDesc = new ActionDescriptor();
            var ref = new ActionReference();
            ref.putEnumerated(charIDToTypeID("Chnl"), charIDToTypeID("Chnl"), charIDToTypeID("Cmps"));
            lvlDesc.putReference(charIDToTypeID("Chnl"), ref);
            var inputList = new ActionList();
            inputList.putInteger(inputBlack);
            inputList.putInteger(inputWhite);
            lvlDesc.putList(charIDToTypeID("Inpt"), inputList);
            lvlDesc.putDouble(charIDToTypeID("Gmm "), gamma);
            var outputList = new ActionList();
            outputList.putInteger(0);
            outputList.putInteger(255);
            lvlDesc.putList(charIDToTypeID("Otpt"), outputList);
            list.putObject(charIDToTypeID("LvlA"), lvlDesc);
            desc.putList(charIDToTypeID("Adjs"), list);
            executeAction(charIDToTypeID("Lvls"), desc, DialogModes.NO);
        } catch (e1) {
            alert("Levels failed on layer: " + doc.activeLayer.name + "\n" + e1);
        }
    }
}

function applyDesaturate() {
    executeAction(charIDToTypeID("Dstt"), undefined, DialogModes.NO);
}

function applyMinimumPX(radius) {
    var desc = new ActionDescriptor();
    desc.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), radius);
    executeAction(charIDToTypeID("Mnm "), desc, DialogModes.NO);
}

function applyMaximumPX(radius) {
    var desc = new ActionDescriptor();
    desc.putUnitDouble(charIDToTypeID("Rds "), charIDToTypeID("#Pxl"), radius);
    executeAction(charIDToTypeID("Mxm "), desc, DialogModes.NO);
}

function mergeActiveLayerDown() {
    try {
        doc.activeLayer.merge();
    } catch (e1) {
        try {
            executeAction(charIDToTypeID("Mrg2"), undefined, DialogModes.NO);
        } catch (e2) {
            alert("Merge failed. Active layer was: " + doc.activeLayer.name + "\n" + e2);
        }
    }
}

function buildMaskFromImg(sourceLayer, settings) {
    doc.activeLayer = sourceLayer;
    var maskLayer = sourceLayer.duplicate();
    maskLayer.name = settings.name;
    doc.activeLayer = maskLayer;

    applyDesaturate();
    applyLevels(settings.levelsBlack, settings.levelsGamma, settings.levelsWhite);
    createThresholdAdjustment(settings.threshold);
    mergeActiveLayerDown();

    doc.activeLayer.name = settings.name;
    maskLayer = doc.activeLayer;

    applyMinimumPX(settings.minimumRadius);
    applyMaximumPX(settings.maximumRadius);

    doc.activeLayer = maskLayer;
    return maskLayer;
}

var imgLayer = doc.activeLayer;
try { imgLayer.isBackgroundLayer = false; } catch (e) {}
imgLayer.name = "img";

var redLayer = doc.artLayers.add();
redLayer.name = "red";
doc.activeLayer = redLayer;
fillRed();
try { redLayer.move(imgLayer, ElementPlacement.PLACEAFTER); } catch (e) {}

var maskHardLayer = buildMaskFromImg(imgLayer, hardSettings);
var maskSoftLayer = buildMaskFromImg(imgLayer, softSettings);

try { maskHardLayer.move(imgLayer, ElementPlacement.PLACEBEFORE); } catch (e) {}
try { maskSoftLayer.move(maskHardLayer, ElementPlacement.PLACEBEFORE); } catch (e) {}

maskSoftLayer.visible = false;
doc.activeLayer = maskHardLayer;
