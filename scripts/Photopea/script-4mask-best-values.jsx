// Photopea / Photoshop JSX
// Start state after reload:
// Background
//
// Final expected stack:
// soft-black
// hard-black
// soft-white
// hard-white
// img
// red
//
// Notes:
// - white masks use Minimum -> Maximum
// - black masks use Maximum -> Minimum
// - values are placeholders; replace them with ROI auto-search results

var doc = app.activeDocument;

//////////////////////////////////////////////////
// SETTINGS
//////////////////////////////////////////////////

var softBlackSettings = { /// BEST
    name: "soft-black",

    // Levels: black / gamma / white
    levelsBlack: 0,
    levelsGamma: 0.80,
    levelsWhite: 50,

    threshold: 8,

    // black background logic: Maximum -> Minimum
    maximumRadius: 2,
    minimumRadius: 2,

    morphOrder: "maxmin"
};

var hardBlackSettings = { /// BEST
    name: "hard-black",

    levelsBlack: 0,
    levelsGamma: 0.50,
    levelsWhite: 40,

    threshold: 12,

    // black background logic: Maximum -> Minimum
    maximumRadius: 3,
    minimumRadius: 3,

    morphOrder: "maxmin"
};

var softWhiteSettings = { /// PLACEHOLDER
    name: "soft-white",

    levelsBlack: 10,
    levelsGamma: 0.60,
    levelsWhite: 110,

    threshold: 64,

    // white background logic: Minimum -> Maximum
    minimumRadius: 10,
    maximumRadius: 10,

    morphOrder: "minmax"
};

var hardWhiteSettings = { /// PLACEHOLDER
    name: "hard-white",

    levelsBlack: 26,
    levelsGamma: 0.40,
    levelsWhite: 110,

    threshold: 32,

    // white background logic: Minimum -> Maximum
    minimumRadius: 14,
    maximumRadius: 14,

    morphOrder: "minmax"
};

//////////////////////////////////////////////////
// HELPERS
//////////////////////////////////////////////////

function fillRed() {
    var redColor = new SolidColor();
    redColor.rgb.red = 255;
    redColor.rgb.green = 0;
    redColor.rgb.blue = 0;

    doc.selection.selectAll();
    doc.selection.fill(redColor);
    doc.selection.deselect();
}

function applyDesaturate() {
    try {
        executeAction(charIDToTypeID("Dstt"), undefined, DialogModes.NO);
    } catch (e) {
        alert("Desaturate failed on layer: " + doc.activeLayer.name + "\n" + e);
    }
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

            ref.putEnumerated(
                charIDToTypeID("Chnl"),
                              charIDToTypeID("Chnl"),
                              charIDToTypeID("Cmps")
            );

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

function createThresholdAdjustment(value) {
    var desc = new ActionDescriptor();
    var ref = new ActionReference();

    ref.putClass(charIDToTypeID("AdjL"));
    desc.putReference(charIDToTypeID("null"), ref);

    var adjDesc = new ActionDescriptor();
    var thresholdDesc = new ActionDescriptor();

    thresholdDesc.putInteger(charIDToTypeID("Lvl "), value);
    adjDesc.putObject(
        charIDToTypeID("Type"),
                      charIDToTypeID("Thrs"),
                      thresholdDesc
    );

    desc.putObject(charIDToTypeID("Usng"), charIDToTypeID("AdjL"), adjDesc);

    executeAction(charIDToTypeID("Mk  "), desc, DialogModes.NO);
}

function applyMinimumPX(radius) {
    if (radius <= 0) {
        return;
    }

    var desc = new ActionDescriptor();
    desc.putUnitDouble(
        charIDToTypeID("Rds "),
                       charIDToTypeID("#Pxl"),
                       radius
    );

    executeAction(charIDToTypeID("Mnm "), desc, DialogModes.NO);
}

function applyMaximumPX(radius) {
    if (radius <= 0) {
        return;
    }

    var desc = new ActionDescriptor();
    desc.putUnitDouble(
        charIDToTypeID("Rds "),
                       charIDToTypeID("#Pxl"),
                       radius
    );

    executeAction(charIDToTypeID("Mxm "), desc, DialogModes.NO);
}

function mergeActiveLayerDown() {
    try {
        doc.activeLayer.merge();
    } catch (e1) {
        try {
            executeAction(charIDToTypeID("Mrg2"), undefined, DialogModes.NO);
        } catch (e2) {
            alert(
                "Merge failed. Active layer was: " +
                doc.activeLayer.name +
                "\n" +
                e2
            );
        }
    }
}

function applyMorphology(settings) {
    if (settings.morphOrder === "maxmin") {
        applyMaximumPX(settings.maximumRadius);
        applyMinimumPX(settings.minimumRadius);
        return;
    }

    if (settings.morphOrder === "minmax") {
        applyMinimumPX(settings.minimumRadius);
        applyMaximumPX(settings.maximumRadius);
        return;
    }

    alert("Unknown morphOrder for " + settings.name + ": " + settings.morphOrder);
}

function buildMaskFromImg(sourceLayer, settings) {
    doc.activeLayer = sourceLayer;

    var maskLayer = sourceLayer.duplicate();
    maskLayer.name = settings.name;
    doc.activeLayer = maskLayer;

    // Stable grayscale pipeline
    applyDesaturate();

    applyLevels(
        settings.levelsBlack,
        settings.levelsGamma,
        settings.levelsWhite
    );

    createThresholdAdjustment(settings.threshold);

    // Merge Threshold adjustment into raster mask layer
    mergeActiveLayerDown();

    doc.activeLayer.name = settings.name;
    maskLayer = doc.activeLayer;

    applyMorphology(settings);

    doc.activeLayer = maskLayer;
    return maskLayer;
}

//////////////////////////////////////////////////
// MAIN
//////////////////////////////////////////////////

var imgLayer = doc.activeLayer;

try {
    imgLayer.isBackgroundLayer = false;
} catch (e) {
    // Continue if Photopea ignores this
}

imgLayer.name = "img";

//////////////////////////////////////////////////
// Create red layer

var redLayer = doc.artLayers.add();
redLayer.name = "red";
doc.activeLayer = redLayer;

fillRed();

try {
    redLayer.move(imgLayer, ElementPlacement.PLACEAFTER);
} catch (e) {
    // Continue if Photopea ignores this
}

//////////////////////////////////////////////////
// Build masks

var hardWhiteLayer = buildMaskFromImg(imgLayer, hardWhiteSettings);
var softWhiteLayer = buildMaskFromImg(imgLayer, softWhiteSettings);
var hardBlackLayer = buildMaskFromImg(imgLayer, hardBlackSettings);
var softBlackLayer = buildMaskFromImg(imgLayer, softBlackSettings);

//////////////////////////////////////////////////
// Force final layer order:
// soft-black
// hard-black
// soft-white
// hard-white
// img
// red

try {
    hardWhiteLayer.move(imgLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

try {
    softWhiteLayer.move(hardWhiteLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

try {
    hardBlackLayer.move(softWhiteLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

try {
    softBlackLayer.move(hardBlackLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

//////////////////////////////////////////////////
// Visibility / active layer

softBlackLayer.visible = false;
hardBlackLayer.visible = true;
softWhiteLayer.visible = false;
hardWhiteLayer.visible = true;

doc.activeLayer = hardBlackLayer;

// Final stack:
// soft-black
// hard-black
// soft-white
// hard-white
// img
// red
