const data = JSON.parse(
  document.getElementById("basevis-entry").dataset.visdata
);

var trace = {
  x: data.X,
  y: data.Y,
  mode: "markers",
  type: "scatter",
  marker: { size: 12 },
  autosize: true,
};

var plot_data = [trace];

var layout = {
  title: "Scatter Plot",
};

Plotly.newPlot("basevis-entry", plot_data, layout, {
  displaylogo: false,
  modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
});
