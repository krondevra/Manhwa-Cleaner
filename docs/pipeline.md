Значительно улучшенный ручной pipeline для максимально быстрой и качественной очистки (вплоть до пикселей). Значения ThresholdValue и softMinMaxRadius тестировались и это лучшее значения. Если двигать значения, то качество начнёт падать. Местами у пары фреймов на рамке есть пару серых пикселей, которые остались. Это можно исправить руками. Основная работа сделана масками. По итогу 3 очистки, 1 заливка. Последние шаги немного не точные.

1. Применить скрипт
```md
// Photopea / Photoshop JSX
// Start state after reload:
// Background
//
// Final expected state:
// mask-soft
// mask-hard
// img
// red

var doc = app.activeDocument;

var hardThresholdValue = 16;
var hardMinMaxRadius = 18;

var softThresholdValue = 141;
var softMinMaxRadius = 6;

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
// 4. Create mask-hard from img

var maskHardLayer = buildMaskFromImg(imgLayer, "mask-hard", hardThresholdValue, hardMinMaxRadius);

//////////////////////////////////////////////////
// 5. Create mask-soft from img

var maskSoftLayer = buildMaskFromImg(imgLayer, "mask-soft", softThresholdValue, softMinMaxRadius);

//////////////////////////////////////////////////
// 6. Force final layer order:
// mask-soft
// mask-hard
// img
// red

try {
    maskHardLayer.move(imgLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

try {
    maskSoftLayer.move(maskHardLayer, ElementPlacement.PLACEBEFORE);
} catch (e) {}

//////////////////////////////////////////////////
// 7. Final active layer

maskSoftLayer.visible = false;
doc.activeLayer = maskHardLayer;
```

2. Слой `mask-hard` -  Используем `Magic Wand (tolerance 0, continues ON, feather 0`), зажав `Shift`, выделяем белый фон между фреймами. 

3. Создаём обтравочную маску на слое `img` с зажатым `Alt`.

4. Слой `mask-soft` -  Используем `Magic Wand (tolerance 0, continues ON, feather 0`), зажав `Shift`, выделяем белый фон между фреймами.

5. Слой `img`, `активная обтравочная маска` - Выбираем `белый цвет`, нажимает `backspace` (заливка).

6. Слой `img`, `активная обтравочная маска` - `Ctrl+Shift+I` (инвертируем текущее выделение), выбираем `чёрный цвет`, зажимаем `Alt` для выделения и выделяем 1 пиксель справа от самого вверха до низа, нажимает `backspace` (заливка).

7. Слой `img`, `активная обтравочная маска` - спользуем `Magic Wand (tolerance 32, continues ON, feather 0`), зажав `Shift`, выделяем белый фон между фреймами (внутри йероглифов тоже). 

8. Слой `img`, `активная обтравочная маска` - `Select -> Modify -> Expand -> 3 px -> OK`, вручную проходимся аккуратно по йероглифам кистью с `белым цветом`.



