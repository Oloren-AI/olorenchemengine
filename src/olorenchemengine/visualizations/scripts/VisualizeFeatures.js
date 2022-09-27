var main = document.getElementById("basevis-entry");
const data = JSON.parse(
    document.getElementById("basevis-entry").dataset.visdata
  );

let options = {};

let smilesDrawer = new SmilesDrawer.Drawer(options);

PROP_DATA = data.PROPERTY_VALUES;

title = document.createElement("h1");
title.innerHTML = "Analyze Property Correlation with Features";
main.appendChild(title);

feature_select = document.createElement("select");
feature_select.id = "feature_select";
feature_select_default = document.createElement("option");
feature_select_default.innerHTML = "Select a feature to analyze against: ";
feature_select_default.value = "";
feature_select_default.disabled = true;
feature_select_default.selected = true;
feature_select.appendChild(feature_select_default);

for (var i = 0; i < data.FEATURE_COLS.length; i++) {
  feature_select_option = document.createElement("option");
  feature_select_option.innerHTML = data.FEATURE_COLS[i] + "; Spearman Rank Correlation: " + data.SPEARMAN_COEF[i];
  feature_select_option.value = i;
  feature_select.appendChild(feature_select_option);
}

main.appendChild(feature_select);

function remove_by_id(id){if (document.contains(document.getElementById(id))) {
  document.getElementById(id).remove();
}}

feature_select.onchange = function() {
    remove_by_id("feature-div")
    remove_by_id("feature-stats")
    remove_by_id("feature-stats2")
    remove_by_id("feature-stats3")

    var i = this.value

    var plot = document.createElement("div");
    plot.id = `feature-div`;
    var stats = document.createElement("p");
    stats.id = `feature-stats`;
    stats.innerHTML = `${data.PROPERTY} vs ${data.FEATURE_COLS[i]} statistics:`;
    var stats2 = document.createElement("p");
    stats2.id = `feature-stats2`;
    stats2.innerHTML = `     Spearman Rank Correlation Coefficient: ${data.SPEARMAN_COEF[i]}`;
    var stats3 = document.createElement("p");
    stats3.id = `feature-stats3`;
    stats3.innerHTML = `     Spearman Correlation p-value: ${data.SPEARMAN_PVAL[i]}`;

    main.appendChild(plot);
    main.appendChild(stats);
    main.appendChild(stats2);
    main.appendChild(stats3);
    var plot_data = [{
        x: data.datacols[data.FEATURE_COLS[i]],
        y: PROP_DATA,
        text: data.SMILES,
        type: 'scatter',
        mode: 'markers',
        hoverinfo: "none",
    }];
    var layout = {
        title: `${data.PROPERTY} vs ${data.FEATURE_COLS[i]}`,
        xaxis: {
          title: data.FEATURE_COLS[i],
        },
        yaxis: {
          title: data.PROPERTY,
        },
      };

    Plotly.newPlot(`feature-div`, plot_data, layout);
    
    const hoverCanvas = document.createElement("canvas");
    hoverCanvas.id = "hoverCanvas";

    const hoverBackground = document.createElement("canvas");
    hoverBackground.id = "hoverBackground";

    var hoverSize =data.hoverSize;

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
        
        x = (d.x - xaxis.range[0]) / (xaxis.range[1] - xaxis.range[0]);
        y = (d.y - yaxis.range[0]) / (yaxis.range[1] - yaxis.range[0]);

        Plotly.update(
          "feature-div",
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
                xanchor: ((x > 0.5) ? 'right' : 'left'),
                yanchor: ((y > 0.5) ? 'top' : 'bottom'),
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
                xanchor: ((x > 0.5) ? 'right' : 'left'),
                yanchor: ((y > 0.5) ? 'top' : 'bottom'),
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
      Plotly.update("feature-div", {}, { annotations: [], images: [] });
    });
}