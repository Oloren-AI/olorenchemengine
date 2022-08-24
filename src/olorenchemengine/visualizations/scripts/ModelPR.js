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
    y: [data.baseline, data.baseline],
    mode: "lines",
    line: {
      dash: "dashdot",
    },
  };
  
  var plot_data = [trace, diag_trace];
  
  var layout = {
    title: data.model_name.concat(" ", "PR curve"),
    showlegend: false,
    xaxis: {
      title: "Recall",
      range: [0, 1],
    },
    yaxis: {
      title: "Precision",
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
        text: "PR-AUC: ".concat(data.score),
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
  