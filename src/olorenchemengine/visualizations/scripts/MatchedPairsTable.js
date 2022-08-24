const data = JSON.parse(
  document.getElementById("basevis-entry").dataset.visdata
);
console.log(data)
div = document.getElementById("basevis-entry");

title = document.createElement("h1");
title.innerHTML = "Find matched pairs";
div.appendChild(title);

desc = document.createElement("p");
desc.innerHTML = "First, select a column to find matched pairs on. Then, select \
  initial residues and final residues to see the chain between two datapoints \
  matched by such residue";
div.appendChild(desc);

col_desc = document.createElement("span");
col_desc.innerHTML = "Select a column: ";
div.appendChild(col_desc);

col_select = document.createElement("select");
col_select.id = "col_select";
col_select_default = document.createElement("option");
col_select_default.innerHTML = "Select a column";
col_select_default.value = "";
col_select_default.disabled = true;
col_select_default.selected = true;
col_select.appendChild(col_select_default);
for (var i = 0; i < data.length; i++) {
  col_select_option = document.createElement("option");
  col_select_option.innerHTML = data[i]["col"];
  col_select_option.value = i;
  col_select.appendChild(col_select_option);
}

div.appendChild(col_select);

function remove_by_id(id){if (document.contains(document.getElementById(id))) {
          document.getElementById(id).remove();
}}

col_select.onchange = function() {
  remove_by_id("initial_span")
  remove_by_id("initial")
  remove_by_id("mp_table_div")

  initial_span = document.createElement("span");
  initial_span.id = "initial_span";
  initial_span.innerHTML = ", Select initial residues: ";
  div.appendChild(initial_span);

  initial = document.createElement("select");
  initial.id = "initial";
  default_option = document.createElement("option");
  default_option.setAttribute("value", "Select an initial residue");
  default_option.innerHTML = "Select an initial residue";
  default_option.disabled = true;
  default_option.selected = true;

  initial.appendChild(default_option);
  initial.initial_residue_idx = this.value;
  unique_initials = data[initial.initial_residue_idx].col_data.map(function(d){return d.initial}).filter((v, i, a) => a.indexOf(v) === i);

  for(var j = 0; j < unique_initials.length; j++) {
      option = document.createElement("option");
      option.setAttribute("value", unique_initials[j]);
      option.innerHTML = unique_initials[j];
      initial.appendChild(option);
  };

  div.appendChild(initial);

  function tabulate(data, columns, selector) {
    var table = d3.select(selector).append('table')
    var thead = table.append('thead')
    var	tbody = table.append('tbody');

    // append the header row
    thead.append('tr')
      .selectAll('th') 
      .data(columns).enter()
      .append('th')
        .text(function (column) { return column; });

    // create a row for each object in the data
    var rows = tbody.selectAll('tr')
      .data(data)
      .enter()
      .append('tr');

    // create a cell in each row for each column
    var cells = rows.selectAll('td')
      .data(function (row) {
        return columns.map(function (column) {
          return {column: column, value: row[column]};
        });
      })
      .enter()
      .append('td')
        .text(function (d) { return d.value; });

    return table;
  }

  initial.onchange = function() {
      mp_table_data = data[initial.initial_residue_idx].col_data.filter(function(d) {
          return d.initial == initial.value;
      });

      remove_by_id("mp_table_div")

      mp_table_div = document.createElement("div");
      mp_table_div.id = "mp_table_div";
      div.appendChild(mp_table_div);
      
      if (mp_table_data.length > 0) {
          tabulate(mp_table_data, Object.keys(mp_table_data[0]).slice(start=1), "#mp_table_div");
      } else {
          mp_table_div.innerHTML = "No matched pairs found";
      }
  }
}