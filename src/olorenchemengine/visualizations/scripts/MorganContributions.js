// set the dimensions and margins of the graph

const data = JSON.parse(document.getElementById('basevis-entry').dataset.visdata);
var plot = document.getElementById('basevis-entry');
let substructures = data.substructures;

var property = data.property;
var original_pred = data.original_prediction;
var compound_id = data.compound_id;



    

let mol;
let isInitialized = false;
let original_svg;
initRDKitModule().then(function (instance) {{
    isInitialized = true;
    RDKitModule = instance;
    console.log("version: " + RDKitModule.version());

        var smiles = data.original_molecule;
        mol = RDKitModule.get_mol(smiles);
        var dest = document.getElementById("basevis-molecule");
        var svg = mol.get_svg();
        svg = svg.replace("width='250px' height='200px'", "width='100%' height='auto'");
        original_svg = svg;
        dest.innerHTML = "<div id='drawing' style='height: 100%; display: flex; flex-direction: column; align-items: center;' >" + svg + "</div>";


    }});


var margin = { top: 10, right: 30, bottom: 30, left: 60 },
  width = 460 - margin.left - margin.right,
  height = 400 - margin.top - margin.bottom;


var trace1 = {
    x: data.maxbits.map((x, i) => data.maxeffect[i] ? "Bit " + x: ""),
    y: data.maxeffect,
    txt: data.minbits,
    name: "Max Effect on " + data.property,
    marker: {color: "#1D62E7"},
    type: 'bar',
};

var trace2 = {
    x: data.minbits.map((x, i) => data.mineffect[i] != undefined ? "Bit " + x: undefined),
    name: "Min Effect on " + data.property,
    y: data.mineffect,
    txt: data.minbits,
    marker: {color: "#6943D8"},
    type: 'bar',
};

var plot_data = [trace1, trace2];

var layout = {
  hovermode:"closest",
  title: "Effect of Morgan Bits on " + data.property + 
  "<br><sub>" + "Original Predicted " + data.property + " for " + data.compound_id + ": " + data.original_prediction.toFixed(3) + "</sub>",
  xaxis: {
    title: 'Bits',
    titlefont: {
        family: 'Cabin, sans-serif',
        size: 18,
        color: '#7f7f7f'
    }
},
yaxis: {
    title: 'Effect on ' + data.property,
    titlefont: {
        family: 'Cabin, sans-serif',
        size: 18,
        color: '#7f7f7f'
    }
},

        
};

let smilesDrawer = new SmilesDrawer.Drawer({ width: 250, height: 250, padding: 15 });

Plotly.newPlot('basevis-entry', plot_data, layout);

let substructure_svg = {}
plot
   .on('plotly_hover', function(data) {
    var bit = data.points.map(function(d){
        
        return (d.x.split(" ")[1]);
    });

    var y = data.points.map(function(d){
        return (d.y);
    });

    var change = (original_pred - y).toFixed(3);
;

        if (substructure_svg[bit] == undefined) {
            
            var smarts = substructures[bit];
            var qmol = RDKitModule.get_qmol(smarts);
            var mdetails = mol.get_substruct_match(qmol);
            
            var svg = mol.get_svg_with_highlights(mdetails);
            svg = svg.replace("width='250px' height='200px'", "width='100%' height='auto'");
            
            var dest = document.getElementById("basevis-molecule");
            substructure_svg
            dest.innerHTML = "<div id='drawing' style='height: 50%; display: flex; flex-direction: column; align-items: center;' >" + svg + "</div>";
            substructure_svg[bit] = svg;
            
            
            
        }
        else {
            var dest = document.getElementById("basevis-molecule");
            dest.innerHTML = "<div id='drawing' style='height: 50%; display: flex; flex-direction: column; align-items: center;' >" + substructure_svg[bit] + "</div>";

        }

        var closeUp = document.createElement("canvas");
        closeUp.setAttribute("id", "closeUp");
        document.getElementById("basevis-molecule").appendChild(closeUp);
        var sub_smiles = substructures[bit];

        SmilesDrawer.parse(sub_smiles, function(tree) {
            // Draw to the canvas
            smilesDrawer.draw(tree, "closeUp", "light", false);
          });
        
       

        var update = {
            title: "Effect of Morgan Bits on " + property + 
                   "<br><sub>" + "Predicted " + property + " with flipped Bit #" + bit +": " + change + "</sub>",
        };
        var graphDiv = document.getElementById('basevis-entry');
        Plotly.relayout(graphDiv, update);

        
      

    } )
    .on('plotly_unhover', function(data) {

        var dest = document.getElementById("basevis-molecule");
        dest.innerHTML = "<div id='drawing' style='height: 100%; display: flex; flex-direction: column; align-items: center;' >" + original_svg + "</div>";

        var update = {
            title: "Effect of Morgan Bits on " + property + 
                   "<br><sub>" + "Original Predicted " + property + " for " + compound_id + ": " + original_pred.toFixed(3) + "</sub>",
        };
        var graphDiv = document.getElementById('basevis-entry');
        Plotly.relayout(graphDiv, update);

    }    
    );

   



