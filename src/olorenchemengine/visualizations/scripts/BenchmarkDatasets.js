const data = JSON.parse(
    document.getElementById("basevis-entry").dataset.visdata
  );
var plot = document.getElementById("basevis-entry");
  
var trace = {
  x: data.model_names,
  y: data.model_counts,
  hovertext: data.model_params,
  hoverlabel: {align: "left"},
  type: "bar"
};

var plot_data = [trace];

var layout = {
  title: "Benchmark Rankings (number of datasets a model is superior on)",
};

function remove_by_id(id){if (document.contains(document.getElementById(id))) {
    document.getElementById(id).remove();
}}

Plotly.newPlot("basevis-entry", plot_data, layout, {
  modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
});
