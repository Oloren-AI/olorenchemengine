const data = JSON.parse(
  document.getElementById("basevis-entry").dataset.visdata
);

var trace = {
  x: data.X,
  y: data.Y,
  mode: "lines",
};

var diag_trace = {
  x: [0, 1],
  y: [0, 1],
  mode: "lines",
  line: {
    dash: "dashdot",
  },
};

var plot_data = [trace, diag_trace];

var layout = {
  title: data.model_name.concat(" ", "ROC curve"),
  showlegend: false,
  xaxis: {
    title: "False Positive Rate",
    range: [0, 1],
  },
  yaxis: {
    title: "True Positive Rate",
    range: [0, 1],
  },
  annotations: [
    {
      xref: "x",
      yref: "y",
      x: 0.05,
      xanchor: "left",
      y: 0.95,
      yanchor: "top",
      text: "ROC-AUC: ".concat(data.score),
      showarrow: false,
      font: {
        color: "#8b0000",
      },
    },
  ],
  autosize: true,
};

Plotly.newPlot("basevis-entry", plot_data, layout, {
  displaylogo: false,
  modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
});
