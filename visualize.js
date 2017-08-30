function _get(url, callback) {
  var q = new XMLHttpRequest();
  q.open('GET', url);
  q.onload = function() {
    callback(JSON.parse(q.responseText));
  }
  q.send();
}

function heatMapColorForValue(value){
  var h = (1.0 - value) * 240;
  return "hsl(" + h + ", 100%, 50%)";
}

// RENDER TABLE (large) -- this is the table format that also displays
// the MSE of each dimension.
function render_table(url, table_id) {
  // Fetch the table
  _get(url, function(data) {

    // Create the table element
    var element = document.createElement('table');
    document.getElementById(table_id).appendChild(element);

    // Get and sort all the model names
    var models = [];
    for (var source in data) {
      models.push(source);
    }
    models.sort();

    // Populate the table with canvases
    var label_row = document.createElement('tr');
    label_row.appendChild(document.createElement('td'));
    for (var i = 0; i < models.length; i += 1) {
      var target = models[i];
      var label_td = document.createElement('td');
      var label_td = document.createElement('td');
      label_td.innerText = target.split('/').reverse()[0].split('.')[0];
      label_row.appendChild(label_td);

      label_td.addEventListener('click', (function(target) {
        return function(event) {
          for (var j = 0; j < models.length; j += 1) {
            rerenderers[models[j]][target]();
          }
        }
      }(target)));
    }
    element.appendChild(label_row);
    element.style.border = '1px solid #000';

    // The rerenderers map.
    // rerenderers[x][y] is the rerendering function sending x to y.
    // rerenderer functions accept a list of values; the indices will be sorted according
    // to the value of that index in the given list. If no list is given, they will be sorted
    // according to their value (so that the graph is monotonically increasing).
    var rerenderers = {}

    for (var q = 0; q < models.length; q += 1) {
      var source = models[q];

      // Each row corresponds to one source
      var row = document.createElement('tr');
      var label_td = document.createElement('td');
      label_td.innerText = source.split('/').reverse()[0].split('.')[0];
      row.appendChild(label_td);

      element.appendChild(row);

      rerenderers[source] = {};

      for (var w = 0; w < models.length; w += 1) {
        var target = models[w];

        // Create a chart out of the sorted data
        var canvas = document.createElement('canvas')
        canvas.width = canvas.height = 100;
        canvas.style.border = '1px solid #000'

        var data_td = document.createElement('td');
        row.appendChild(data_td);

        data_td.appendChild(canvas);

        // Don't worry about comparing models to themselves
        if (q === w) continue;

        // Sort the data
        var mses = data[source][target][0];

        var ctx = canvas.getContext('2d');

        var zipped_mses = mses.map(function(x, i) { return [x, i] });

        // Give each canvas an alt-text equal to the geometric mean of the dimensionwise MSEs
        canvas.title = Math.pow(Math.E, (mses.map(Math.log).reduce(function(a, b) { return a + b;}) / 500)).toString()

        // THE RERENDERING FUNCTION
        // ========================
        //
        // Wrapped in a function for closure reasons
        rerenderers[source][target] = (function(canvas, ctx, zipped_mses, mses) {

          // Closure the geometric mean
          var overall_numeric_mean = Math.pow(Math.E, (mses.map(Math.log).reduce(function(a, b) { return a + b;}) / 500))
          var overall_mean = overall_numeric_mean.toPrecision(3);

          return function rerender_sortby(list) {
            list = list || mses;

            var sorted_mses;
            if (list) {
              // Sort the mses acording to the given sort list
              sorted_mses = zipped_mses.sort(function(a, b) {
                if (list[a[1]] > list[b[1]]) {
                  return 1;
                }
                else if (list[a[1]] < list[b[1]]) {
                  return -1;
                }
                else {
                  return 0;
                }
              }).map(function(l) { return l[0]; });

            }

            else {
              sorted_mses = mses;
            }

            // Clear
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw the graph
            ctx.beginPath();
            ctx.moveTo(0, canvas.height);
            for (var i = 0; i < mses.length; i += 1) {
              ctx.lineTo(i / 500 * canvas.width, (1 - sorted_mses[i]) * canvas.height);
            }

            ctx.lineTo(canvas.width, canvas.height);
            ctx.strokeStyle = ctx.fillStyle = '#000';
            ctx.stroke();
            ctx.fill();

            // Write the geometric average on a heatmap-colored
            // box in the middle of the graph
            var width = ctx.measureText(overall_mean).width
            ctx.fillStyle = heatMapColorForValue(overall_numeric_mean); //'#FFF';
            ctx.fillRect((canvas.width - width) / 2 - 2.5, canvas.height - 33, width + 5, 15);

            ctx.fillStyle = '#000';
            ctx.fillText(overall_mean, (canvas.width - width) / 2, canvas.height - 20);
          };

        }(canvas, ctx, zipped_mses, mses));

        // Currently sort by yourself.
        rerenderers[source][target](mses);

        // On click, sort everyone else by yourself.
        canvas.addEventListener('click', (function(target, mses) {
          return function(event) {
            for (var p = 0; p < models.length; p += 1) {
              var source = models[p];
              rerenderers[source][target](mses);
            }
          };
        }(target, mses)));
      }
    }
  });
}

// RENDER TABLE (small)
// For inclusion in papers, posters, etc.; renders a matrix of heatmap-colored squares.
//
// Uses differential entropy.
function render_small_table(url, table_id) {
  _get(url, function(data) {
    // RENDERING THE TABLE

    // Get model names
    var models = [];
    for (var source in data) {
      models.push(source);
    }
    models = models.filter(function(name) {return name.indexOf('en-en') == -1;});
    models.sort();

    // Get the total minimum and maximum differential entropies
    var total_min = Infinity, total_max = -Infinity;
    for (var i = 0; i < models.length; i += 1) {
      for (var j = 0; j < models.length; j += 1) {
        if (j != i) {
          var source = models[j];

          var mses = data[source][target][0];

          var overall_numeric_mean = mses.map(function(x) { return Math.log(Math.sqrt(x * 2 * Math.PI * Math.E)); }).reduce(function(a, b) { return a + b;});

          total_min = Math.min(total_min, overall_numeric_mean);
          total_max = Math.max(total_max, overall_numeric_mean);
        }
      }
    }

    // Create the canvas
    var small_table_canvas = document.createElement('canvas');
    document.getElementById(table_id).appendChild(small_table_canvas);
    small_table_canvas.width = small_table_canvas.height = 500;

    // Make space for the legend
    var legend_height = 30;

    // Make space for the axis labels
    var xmargin = ymargin = 25;

    // Get the resultant graph dimensions
    var graph_height = small_table_canvas.height - ymargin - legend_height;
    var graph_width = small_table_canvas.height - xmargin;
    var small_table_ctx = small_table_canvas.getContext('2d');

    // Short name turns `/asdf/asdf/en-es-0.t7` into `es-0`.
    function shorten_name(x) {
      return x.split('/').reverse()[0].split('.')[0].substr(3, 4);
    }

    // Iterate over the pairs and fill the correct square.
    for (var i = 0; i < models.length; i += 1) {
      for (var j = 0; j < models.length; j += 1) {
        if (i == j) continue;
        var source = models[i], target=models[j];

        var mses = data[source][target][0];
        var overall_numeric_mean = mses.map(function(x) { return Math.log(Math.sqrt(x * 2 * Math.PI * Math.E)); }).reduce(function(a, b) { return a + b;});
        small_table_ctx.fillStyle = heatMapColorForValue((overall_numeric_mean - total_min) / (total_max - total_min));

        // Source is vertical position,
        // target horizontal
        small_table_ctx.fillRect(
          j / models.length * graph_width + xmargin, i / models.length * graph_height + ymargin,
          graph_width / models.length, graph_height / models.length
        );
      }

      // Also print the name of this model in the axis space
      small_table_ctx.fillStyle = '#000';
      small_table_ctx.textBaseline = 'middle';
      small_table_ctx.fillText(shorten_name(models[i]), 0, (i + 0.5) / models.length * graph_height + ymargin);

      small_table_ctx.save();
      small_table_ctx.translate((i + 0.5) / models.length * graph_width + xmargin, ymargin);
      small_table_ctx.rotate(-Math.PI / 2);

      small_table_ctx.fillStyle = '#000';
      small_table_ctx.textBaseline = 'middle';
      small_table_ctx.fillText(shorten_name(models[i]), 0, 0);

      small_table_ctx.restore();
    }

    // Draw legend
    var LEGEND_WIDTH = 200;
    var canvas_center = small_table_canvas.width / 2;
    var left_side = canvas_center - LEGEND_WIDTH / 2 + xmargin;

    for (var i = 0; i < LEGEND_WIDTH; i+= 1) {
      var heatmap_value = (i / LEGEND_WIDTH) * (total_max - total_min) + total_min;
      small_table_ctx.fillStyle = heatMapColorForValue(i / LEGEND_WIDTH);
      small_table_ctx.fillRect(left_side + i, graph_height + ymargin + 5, 1, legend_height - 5);
    }
    small_table_ctx.fillStyle = '#000';
    small_table_ctx.font = '15px Arial';
    small_table_ctx.textBaseline = 'middle';

    var total_min_text = total_min.toPrecision(3) + ' nats';
    small_table_ctx.fillText(total_min_text, left_side - small_table_ctx.measureText(total_min_text.toString()).width - 5, graph_height + ymargin + 17.5)

    small_table_canvas.title = url;

    var total_max_text = total_max.toPrecision(3) + ' nats';
    small_table_ctx.fillText(total_max_text, left_side + LEGEND_WIDTH + 5, graph_height + ymargin + 17.5)
  });
}
