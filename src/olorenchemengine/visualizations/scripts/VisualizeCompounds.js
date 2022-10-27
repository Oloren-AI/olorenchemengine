const data = JSON.parse(
    document.getElementById("basevis-entry").dataset.visdata
  );

let options = {"width": data.compound_width, "height": data.compound_height};
let smilesDrawer = new SmilesDrawer.Drawer(options);

basevis = document.getElementById("basevis-entry");
table = document.createElement("table");
basevis.appendChild(table);

for (var y = 0; y < data.table_height; y++) {
  let tr = document.createElement("tr");
  table.appendChild(tr);
  for (var x = 0; x < data.table_width; x++) {
    let ix = y*data.table_width + x;
    let td = document.createElement("td");
    let div = document.createElement("div");
    let canvas = document.createElement("canvas");
    canvas.id = `canvas_${x}-${y}`;
    div.appendChild(canvas);
    if ("annotations" in data) {
      for (const [key, value] of Object.entries(data.annotations)) {
        let span = document.createElement("SPAN");
        span.innerHTML = `${key}: ${value[ix]};`;
        div.appendChild(span);
      }
    }
    if (data.box) {
      div.style.border = "1px solid black";
    }
    td.appendChild(div);
    tr.appendChild(td);

    SmilesDrawer.parse(data.smiles[ix], function (smiles) {
      smilesDrawer.draw(smiles, `canvas_${x}-${y}`, "light", false, highlights = data["highlights"]);
    });
  }
}