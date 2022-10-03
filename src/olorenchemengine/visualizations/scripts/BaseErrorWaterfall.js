const data = JSON.parse(
    document.getElementById("basevis-entry").dataset.visdata
  );
  
  var trace = {
    x: data.model_names,
    y: data.diffs,
    type: "waterfall",
    textposition: "outside",
    text: data.text,
    measure: data.w_type,
    orientation: "v",
    autosize: true,
    xhoverformat: data.model_names,
    increasing: {marker: {color: "#1F3363"}},
    decreasing: {marker: {color: "#631A43"}},
    totals: {marker: {color: "#4065B0"}},
    connector: {line: {color: "rgb(54, 54, 255)"}},
  };
  
  var plot_data = [trace];
  
  var layout = {
    title: {
        text: "Base Boosted Model Waterfall Plot Residuals (Standard Deviation)",
    },
    xaxis: {
        title: {text: "Model Iterations"},
        type: "category"
    },
    yaxis: {
        title: {text: "Standard Deviation"},
        type: "linear"
    },
  };
  
  Plotly.newPlot("basevis-entry", plot_data, layout, {
    displaylogo: false,
    modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
  });
  