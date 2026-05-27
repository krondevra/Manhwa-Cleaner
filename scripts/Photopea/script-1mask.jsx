// Photopea / Photoshop JSX
// Start state after reload:
// Background
//
// Final expected state:
// mask
// img
// red

var doc = app.activeDocument;

var thresholdValue = 64;
var minMaxRadius = 3;

//////////////////////////////////////////////////
// Helpers

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

//////////////////////////////////////////////////
// 1. Start from current Background layer

var imgLayer = doc.activeLayer;

// Try to unlock Background
try {
    imgLayer.isBackgroundLayer = false;
} catch (e) {
    // Continue if Photopea ignores this
}

// Rename original layer
imgLayer.name = "img";

//////////////////////////////////////////////////
// 2. Create red layer and fill it

var redLayer = doc.artLayers.add();
redLayer.name = "red";
doc.activeLayer = redLayer;

fillRed();

//////////////////////////////////////////////////
// 3. Move red below img

try {
    redLayer.move(imgLayer, ElementPlacement.PLACEAFTER);
} catch (e) {
    // If Photopea ignores this, continue
}

//////////////////////////////////////////////////
// 4. Duplicate img and name duplicate mask

doc.activeLayer = imgLayer;

var maskLayer = imgLayer.duplicate();
maskLayer.name = "mask";
doc.activeLayer = maskLayer;

//////////////////////////////////////////////////
// Current expected stack:
// mask
// img
// red

//////////////////////////////////////////////////
// 5. Create Threshold 64 above mask

createThresholdAdjustment(thresholdValue);

// IMPORTANT:
// Right now active layer should be the new Threshold adjustment layer.

//////////////////////////////////////////////////
// 6. Merge active Threshold layer down into mask

try {
    doc.activeLayer.merge();
} catch (e1) {
    try {
        executeAction(charIDToTypeID("Mrg2"), undefined, DialogModes.NO);
    } catch (e2) {
        alert("Merge failed. Active layer after creating Threshold was: " + doc.activeLayer.name);
    }
}

// After merge, active layer should be the merged black-and-white mask
doc.activeLayer.name = "mask";
maskLayer = doc.activeLayer;

//////////////////////////////////////////////////
// 7. Apply Minimum and Maximum with 3 px to mask

doc.activeLayer = maskLayer;

applyMinimumPX(minMaxRadius);
applyMaximumPX(minMaxRadius);

//////////////////////////////////////////////////
// 8. Final active layer

doc.activeLayer = maskLayer;

// Final expected stack:
// mask
// img
// red
