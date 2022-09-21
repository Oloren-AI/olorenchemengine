var div = document.getElementById("basevis-entry");
var data = JSON.parse(document.getElementById("basevis-entry").dataset.visdata);

var title = document.createElement("h2");
title.innerHTML = "Sensitivity of the prediction to the input features";
div.appendChild(title)

var begin_span = document.createElement("span");
begin_span.innerHTML = "more sensitive ";
div.appendChild(begin_span);

for (var i = 0; i < data["highlights"].length; i++) {
  var dot = document.createElement("span");
  dot.style.height = "10px";
  dot.style.width = "10px";
  dot.style.backgroundColor = data["highlights"][i][1];
  dot.style.borderRadius = "50%";
  dot.style.display = "inline-block";
  div.appendChild(dot);
  
  var spacer = document.createElement("span");
  spacer.innerHTML = " ";
  div.appendChild(spacer);
}

var end_span = document.createElement("span");
end_span.innerHTML = "most sensitive";
div.appendChild(end_span);

var div2 = document.createElement("div");
div.appendChild(div2);

var canvas = document.createElement("canvas");
canvas.id = "smiles-canvas";
div2.appendChild(canvas);

let options = {};
let smilesDrawer = new SmilesDrawer.Drawer(options);

SmilesDrawer.parse(data["SMILES"], function(smi) {
    // Draw to the canvas
    smilesDrawer.draw(smi, "smiles-canvas", "light", false, highlights=data["highlights"]);
  });