var plot = document.getElementById("basevis-entry");
var hoverInfo = document.getElementById("basevis-entry2");

var data = JSON.parse(document.getElementById("basevis-entry").dataset.visdata);

let options = {};
let smilesDrawer = new SmilesDrawer.Drawer(options);

var trace = {
  x: data.X,
  y: data.Y,
  text: data.SMILES,
  mode: "markers",
  type: "scatter",
  hoverinfo: "none",
  marker: {},
};

if ("hovertext" in data) {
  trace.hovertext = data.hovertext;
  trace.hoverinfo = "text";
}

if ("group" in data) {
  trace.transforms = [{ type: "groupby", groups: data.group }];
}

if ("hovertemplate" in data) {
  trace.hovertemplate = data.hovertemplate;
}

if ("color" in data) {
  trace.marker.color = data.color;
}

if ("size" in data) {
  trace.marker.size = data.size;
} else {
  trace.marker.size = 12;
}

if ("outline" in data) {
  trace.marker.line = {
    color: data.outline,
    width: 4,
  };
}

if ("opacity" in data) {
  trace.marker.opacity = data.opacity;
}

var plot_data = [trace];

var layout = {
  title: "Plot Title",
  xaxis: {
    title: "x Axis",
  },
  yaxis: {
    title: "y Axis",
  },
  autosize: true,
};

if ("title" in data) {
  layout.title = data.title;
}

if ("xaxis_title" in data) {
  layout.xaxis.title = data.xaxis_title;
}

if ("yaxis_title" in data) {
  layout.yaxis.title = data.yaxis_title;
}

layout.xaxis.type = data["xaxis_type"];
layout.yaxis.type = data["yaxis_type"];

if ("ydomain" in data){
  layout.yaxis.domain = data.ydomain;
}
if ("xdomain" in data){
  layout.xaxis.domain = data.xdomain;
}

if ("yrange" in data){
  layout.yaxis.range = data.yrange;
}
if ("xrange" in data){
  layout.xaxis.range = data.xrange;
}

if ("axesratio" in data){
  layout.yaxis.scaleanchor = "x";
  layout.yaxis.scaleratio = data.axesratio;
}

if ("xdtick" in data){
  layout.xaxis.dtick = data.xdtick;
}

if ("ydtick" in data){
  layout.yaxis.dtick = data.ydtick;
}

if ("color" in data) {
  if ("colorscale" in data) {
    trace.colorscale = data.colorscale;
    trace.marker.colorscale = data.colorscale;
    trace.marker.colorbar = {
      title: "",
      titleside: "top",
    };
  }
}

Plotly.newPlot(
  "basevis-entry",
  plot_data,
  layout,
  (config = {
    displaylogo: false,
    modeBarButtonsToRemove: ["zoom2d", "pan2d", "select2d", "lasso2d"],
  })
);

if ("layout_update" in data){
  Plotly.update("basevis-entry", {}, data.layout_update);
}

if ("trace_update" in data){
  Plotly.addTraces("basevis-entry", data.trace_update);
}

const hoverCanvas = document.createElement("canvas");
hoverCanvas.id = "hoverCanvas";

const hoverBackground = document.createElement("canvas");
hoverBackground.id = "hoverBackground";

var hoverSize = parseFloat(data.hover_size);

plot.appendChild(hoverCanvas);
plot.appendChild(hoverBackground);

plot
  .on("plotly_hover", function (data) {
    var xaxis = data.points[0].xaxis,
      yaxis = data.points[0].yaxis;

    data.points.map(function (d) {
      hoverCanvas.hidden = false;
      hoverBackground.hidden = false;

      ctx = hoverCanvas.getContext("2d");

      SmilesDrawer.parse(d.text, function (smiles) {
        smilesDrawer.draw(smiles, "hoverCanvas", "light", false);
      });

      ctx.fillStyle = "black";
      ctx.shadowColor = "black";
      ctx.shadowBlur = 20;
      ctx.lineJoin = "bevel";
      ctx.lineWidth = 15;
      ctx.strokeRect(0, 0, hoverCanvas.width, hoverCanvas.height, "light");

      hoverBackground.width = hoverCanvas.width;
      hoverBackground.height = hoverCanvas.height;
      ctx = hoverBackground.getContext("2d");
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, hoverBackground.width, hoverBackground.height);

      var x = (d.x - xaxis.range[0]) / (xaxis.range[1] - xaxis.range[0]);
      var y = (d.y - yaxis.range[0]) / (yaxis.range[1] - yaxis.range[0]);
      var xdir = ((x > 0.5) ? 'right' : 'left');
      var ydir = ((y > 0.5) ? 'top' : 'bottom');
      Plotly.update(
        "basevis-entry",
        {},
        {
          // margin: { t: 100, b: 100, l: 50, r: 50 },
          annotations: [
            {
              x: d.x,
              y: d.y,
              xref: "x",
              yref: "y",
              text: "",
              showarrow: true,
              arrowhead: 7,
              ax: 0,
              ay: 0,
            },
          ],
          images: [
            {
              x: x,
              y: y,
              sizex: hoverSize,
              sizey: hoverSize,
              xref: "paper",
              yref: "paper",
              source: document.getElementById("hoverBackground").toDataURL(),
              xanchor: xdir,
              yanchor: ydir,
              layer: "above",
            },
            {
              x: x,
              y: y,
              sizex: hoverSize,
              sizey: hoverSize,
              xref: "paper",
              yref: "paper",
              source: document.getElementById("hoverCanvas").toDataURL(),
              xanchor: xdir,
              yanchor: ydir,
              layer: "above",
            },
          ],
        }
      );
      hoverCanvas.hidden = true;
      hoverBackground.hidden = true;
    });
  })
  .on("plotly_unhover", function (data) {
    Plotly.update("basevis-entry", {}, { annotations: [], images: [] });
  });
