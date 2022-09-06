const visdata = JSON.parse(
    document.getElementById("basevis-entry").dataset.visdata
  );
  var basevis = document.getElementById("basevis-entry")
  basevis.style.setProperty("display", "grid") 
  basevis.style.setProperty("grid-template-columns", "2fr  1fr") 
  basevis.style.setProperty("grid-template-rows", "2fr  2fr") 
  basevis.style.setProperty("gap", "2px") 
  basevis.style.setProperty("padding", "2px") 
  var plot = document.createElement("div")
  plot.style.setProperty("grid-row-start", 1)
  plot.style.setProperty("grid-row-end", 3)
  plot.id = "plot"
  var pie1 = document.createElement("div")
  pie1.id = "pie1"
  var pie2 = document.createElement("div")
  pie2.id = "pie2"
  
  basevis.appendChild(plot)
  basevis.appendChild(pie1)
  basevis.appendChild(pie2)
  
  var trace = {
    x: visdata.X,
    y: visdata.Y,
    hovertemplate: '<b>TPR</b>: %{y:.2f}' +
      '; <b>FPR</b>: %{x:.2f}',
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
    title: visdata.model_name.concat(" ", "ROC curve"),
    height: 400,
    width: 400,
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
        text: "ROC-AUC: ".concat(visdata.score),
        showarrow: false,
        font: {
          color: "#8b0000",
        },
      },
    ],
    autosize: true,
  };
  
  Plotly.newPlot("plot", plot_data, layout, {
    displaylogo: false,
    modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
  });
  
  plot.on('plotly_hover', function(d){
    console.log(visdata.P)
    console.log(visdata.N)
    console.log(d.xvals[0])
    console.log(d.yvals[0])
    console.log(d)
  
    var data1 = [{
      values: [visdata.P * d.yvals[0], visdata.N * d.xvals[0]],
      labels: ['True Positive', 'False Positive'],
      type: 'pie',
      textinfo: "label+percent",
      textposition: "outside",
      automargin:true
    }];
    
    var layout1 = {
      title: "Predicted Positives",
      height: 300,
      width: 300,
      showlegend: false
    };
    console.log("pie")
    Plotly.newPlot('pie1', data1, layout1);
  
  var data2 = [{
    values: [visdata.P * (1-d.yvals[0]), visdata.N * (1-d.xvals[0])],
    labels: ['True Positive', 'False Positive'],
    type: 'pie',
    textinfo: "label+percent",
    textposition: "outside",
    automargin:true
  }];
  
  var layout2 = {
    title: "Predicted Negatives",
    height: 300,
    width: 300,
    showlegend: false
  };
  Plotly.newPlot('pie2', data2, layout2);})
  
  .on('plotly_unhover', function(d){
    Plotly.purge('pie1');
    Plotly.purge('pie2');
  });