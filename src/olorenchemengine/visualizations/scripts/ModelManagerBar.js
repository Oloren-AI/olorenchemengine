// set the dimensions and margins of the graph
var margin = { top: 10, right: 30, bottom: 30, left: 60 },
  width = 460 - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;

  const data = JSON.parse(document.getElementById('basevis-entry').dataset.visdata);

var trace = {
  x: data["Model Name"],
  y: data[data.metric],
  type: "bar",
};

var plot_data = [trace];

var layout = {
  title: data.title,
};

Plotly.newPlot("basevis-entry", plot_data, layout);
