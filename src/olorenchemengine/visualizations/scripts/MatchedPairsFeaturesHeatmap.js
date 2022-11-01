const data = JSON.parse(
  document.getElementById("basevis-entry").dataset.visdata
);
console.log(data)
div = document.getElementById("basevis-entry");

for (var i = 0; i < data.length; i++) {
  entry = data[i];

  entry_div = document.createElement("div");
  entry_div.id = `#heatmap${i}`
  div.appendChild(entry_div);

  var entry_data = [
    {
      z: entry.z,
      x: entry.x,
      y: entry.y,
      type: "heatmap",
      hoverongaps: false
    }
  ];
  
  var layout_data = {
    title: entry.title
  };

  Plotly.newPlot(`#heatmap${i}` , entry_data, layout_data);
}