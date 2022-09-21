var div = document.getElementById("basevis-entry");
var data = JSON.parse(document.getElementById("basevis-entry").dataset.visdata);

var canvas = document.createElement("canvas");
canvas.id = "smiles-canvas";
div.appendChild(canvas);

let options = {};
let smilesDrawer = new SmilesDrawer.Drawer(options);

SmilesDrawer.parse(data["SMILES"], function(smi) {
    // Draw to the canvas
    smilesDrawer.draw(smi, "smiles-canvas", "light", false, highlights=data["highlights"]);
  });