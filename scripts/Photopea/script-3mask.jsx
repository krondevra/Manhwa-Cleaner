// Photopea / Photoshop JSX
// Start state after reload:
// Background
//
// Final expected state:
// mask-3
// mask-2
// mask-1
// img
// red

var doc = app.activeDocument;

var mask3ThresholdValue = 80;
var mask3MinMaxRadius = 4;

var mask2ThresholdValue = 32;
var mask2MinMaxRadius = 4;

var mask1ThresholdValue = 13;
var mask1MinMaxRadius = 18;


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

function mergeActiveLayerDown() {
    try {
        doc.activeLayer.merge();
    } catch (e1) {
        try {
            executeAction(charIDToTypeID("Mrg2"), undefined, DialogModes.NO);
        } catch (e2) {
            alert("Merge failed. Active layer was: " + doc.activeLayer.name);
        }
    }
}

function buildMaskFromImg(sourceLayer, maskName, thresholdValue, minMaxRadius) {
    doc.activeLayer = sourceLayer;

    var newMaskLayer = sourceLayer.duplicate();
    newMaskLayer.name = maskName;
    doc.activeLayer = newMaskLayer;

    // Create threshold adjustment above duplicated layer
    createThresholdAdjustment(thresholdValue);

    // Merge threshold layer down into duplicated layer
    mergeActiveLayerDown();

    // After merge, active layer should be the rasterized black-and-white layer
    doc.activeLayer.name = maskName;
    newMaskLayer = doc.activeLayer;

    // Apply Minimum and Maximum
    applyMinimumPX(minMaxRadius);
    applyMaximumPX(minMaxRadius);

    doc.activeLayer = newMaskLayer;
    return newMaskLayer;
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
    // Continue if Photopea ignores this
}

//////////////////////////////////////////////////
// 4. Create mask-1 from img

var mask1Layer = buildMaskFromImg(imgLayer, "mask-1", mask1ThresholdValue, mask1MinMaxRadius);

//////////////////////////////////////////////////
// 5. Create mask-2 from img

var mask2Layer = buildMaskFromImg(imgLayer, "mask-2", mask2ThresholdValue, mask2MinMaxRadius);

//////////////////////////////////////////////////
// 6. Create mask-3 from img

var mask3Layer = buildMaskFromImg(imgLayer, "mask-3", mask3ThresholdValue, mask3MinMaxRadius);

//////////////////////////////////////////////////
// 7. Force final layer order:
// mask-3
// mask-2
// mask-1
// img
// red

try {
    mask1Layer.move(imgLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

try {
    mask2Layer.move(mask1Layer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

try {
    mask3Layer.move(mask2Layer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

//////////////////////////////////////////////////
// 8. Final active layer

mask3Layer.visible = false;
mask2Layer.visible = false;
doc.activeLayer = mask1Layer;
