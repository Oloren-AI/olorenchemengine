const data = JSON.parse(
  document.getElementById("basevis-entry").dataset.visdata
);

var trace = {
  x: data.shap_index,
  y: data.shap_values,
  type: 'bar',
  autosize: true,
};

var plot_data = [trace];

var layout = {
  title: {
      text: "Top 10 Feature Shapley Values"
  },
  xaxis: {
      title: {text: "Vector Indices"},
      type: "linear"
  },
  yaxis: {
      title: {text: "Shapley Value"},
      type: "linear"
  },
};


Plotly.newPlot("basevis-entry", plot_data, layout, {
  displaylogo: false,
  modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
});

