var plot = document.getElementById("basevis-entry");
var data = JSON.parse(document.getElementById("basevis-entry").dataset.visdata);

var trace = {
    hoverinfo: "x",
    hoveron: "points+kde",
    points: "skip",
    pointpos: -0.05,
    box: {},
    jitter: 0,
    name: "Training Distribution",
    showlegend: true,
    marker: {
        line: {
            width: 0.1,
            color: "#8dd3c7"
        },
        size: 10,
        symbol: "line-ns"
    },
    side: "positive",
    type: "violin",
    line: {
        color: "#8dd3c7"
    },
    y0: " ",
    x: data.reference,
    orientation: "h"
}

var trace2 = {
    hoverinfo: "skip",
    name: "Predicted Value and CI",
    showlegend: true,
    side: "positive",
    type: "violin",
    visible: "legendonly",
    line: {
        color: "#bebada",
    },
    y0: " ",
    x: [data.value],
    orientation: "h",
    span: [
        data.value - data.error, data.value + data.error
    ],
}

if ("ci" in data) {
    trace2.name = `Predicted Value and ${data.ci}% CI`
}

if ("box" in data) {
    trace.box.visible = true
}

if ("points" in data) {
    trace.points = "all"
}

var plot_data = [trace, trace2]
var layout = {
    shapes: [
        {
            type: 'rect',
            xref: 'x',
            yref: 'paper',
            x0: data.value - data.error,
            y0: 0,
            x1: data.value + data.error,
            y1: 1,
            fillcolor: '#bebada',
            opacity: 0.5,
            line: {
                width: 0
            }
        },
        {
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: data.value,
            y0: 0,
            x1: data.value,
            y1: 1,
            line: {
                color: '#bebada',
                width: 3
            }
        },
    ],
    title: data.title,
    xaxis: {
        title: data.xaxis_title,
        zeroline: false
    },
    yaxis: {
        title: data.yaxis_title,
    },
    autosize: true,
    showlegend: true,
    legend: {
        x: 0.5,
        y: 0
    }
};

Plotly.newPlot("basevis-entry", plot_data, layout)
