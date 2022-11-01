var div = document.getElementById("basevis-entry");
var data = JSON.parse(document.getElementById("basevis-entry").dataset.visdata);

var title = document.createElement("h2");
title.innerHTML = "Compound Distance-based Matched Pairs";
div.appendChild(title)

col_desc = document.createElement("span");
col_desc.innerHTML = "Select a pair: ";
div.appendChild(col_desc);

col_select = document.createElement("select");
col_select.id = "col_select";
col_select_default = document.createElement("option");
col_select_default.innerHTML = "Select a pair";
col_select_default.value = "";
col_select_default.disabled = true;
col_select_default.selected = true;
col_select.appendChild(col_select_default);

for (var i = 0; i < data["ids"].length; i++) {
  col_select_option = document.createElement("option");
  col_select_option.innerHTML = data["ids"][i] + data["table"][i];
  col_select_option.value = i;
  col_select.appendChild(col_select_option);
}

div.appendChild(col_select);

function remove_by_id(id){if (document.contains(document.getElementById(id))) {
          document.getElementById(id).remove();
}}

var div2 = document.createElement("div");
div.appendChild(div2);
col_select.onchange = function() {
    remove_by_id("smiles-canvas-1");
    remove_by_id("smiles-canvas-2");

    pair_data = data["table"][this.value];

    for (var i = 0; i < data["annotations"].length; i++) {
        remove_by_id("annotation-" + i);
        var annotation = document.createElement("p");
        annotation.id = "annotation-" + i;
        annotation.innerHTML = data["annotations"][i] + ": " + pair_data[data["annotations"][i]];
        div2.appendChild(annotation);
    }

    var canvas1 = document.createElement("canvas");
    canvas1.id = "smiles-canvas-1";
    div2.appendChild(canvas1);
    var canvas2 = document.createElement("canvas");
    canvas2.id = "smiles-canvas-2";
    div2.appendChild(canvas2);
    
    let options = {};
    let smilesDrawer = new SmilesDrawer.Drawer(options);
    
    SmilesDrawer.parse(pair_data["display_smiles_1"], function(smi) {
        // Draw to the canvas
        smilesDrawer.draw(smi, "smiles-canvas-1", "light", false, highlights=data["highlights"]);
      });
    
    SmilesDrawer.parse(pair_data["display_smiles_2"], function(smi) {
      // Draw to the canvas
      smilesDrawer.draw(smi, "smiles-canvas-2", "light", false, highlights=data["highlights"]);
    });
}