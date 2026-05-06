(function () {
  function buildLineDataset(label, data, color, fill) {
    return {
      label: label,
      data: data,
      borderColor: color,
      backgroundColor: fill || color,
      tension: 0.25,
      pointRadius: 3,
      pointHoverRadius: 4,
    };
  }

  function destroyExistingChart(canvas) {
    if (canvas && canvas._chartInstance) {
      canvas._chartInstance.destroy();
    }
  }

  function createChart(canvasId, config) {
    var canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') {
      return;
    }
    destroyExistingChart(canvas);
    canvas._chartInstance = new Chart(canvas, config);
  }

  function renderPrecisionChart(canvasId, precisionChart) {
    createChart(canvasId, {
      type: 'line',
      data: {
        labels: precisionChart.labels,
        datasets: [
          buildLineDataset('train', precisionChart.datasets[0].precision, '#b45309'),
          buildLineDataset('val', precisionChart.datasets[1].precision, '#0f766e'),
          buildLineDataset('test', precisionChart.datasets[2].precision, '#1d4ed8'),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom' },
        },
        scales: {
          y: { beginAtZero: true, suggestedMax: 1 },
        },
      },
    });
  }

  function renderFeatureChart(canvasId, featureChart) {
    createChart(canvasId, {
      type: 'bar',
      data: {
        labels: featureChart.labels,
        datasets: [
          {
            label: 'importance',
            data: featureChart.values,
            backgroundColor: '#0f766e',
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
      },
    });
  }

  function renderEventChart(canvasId, eventChart, title) {
    createChart(canvasId, {
      type: 'bar',
      data: {
        labels: eventChart.event_names,
        datasets: [
          {
            label: 'event rate',
            data: eventChart.event_rate,
            backgroundColor: '#c2410c',
          },
          {
            label: 'precision @ 0.5%',
            data: eventChart.precision_at_0_5,
            backgroundColor: '#0f766e',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: title },
          legend: { position: 'bottom' },
        },
        scales: {
          y: { beginAtZero: true, suggestedMax: 1 },
        },
      },
    });
  }

  function pickFold(chartData, foldName) {
    return chartData.datasets.find(function (dataset) {
      return dataset.label === foldName;
    });
  }

  function renderComparePrecisionChart(canvasId, leftChart, rightChart, leftLabel, rightLabel) {
    var leftVal = pickFold(leftChart, 'val') || { precision: [] };
    var leftTest = pickFold(leftChart, 'test') || { precision: [] };
    var rightVal = pickFold(rightChart, 'val') || { precision: [] };
    var rightTest = pickFold(rightChart, 'test') || { precision: [] };

    createChart(canvasId, {
      type: 'line',
      data: {
        labels: leftChart.labels.length ? leftChart.labels : rightChart.labels,
        datasets: [
          buildLineDataset(leftLabel + ' val', leftVal.precision, '#0f766e'),
          buildLineDataset(leftLabel + ' test', leftTest.precision, '#1d4ed8'),
          buildLineDataset(rightLabel + ' val', rightVal.precision, '#b45309'),
          buildLineDataset(rightLabel + ' test', rightTest.precision, '#7c3aed'),
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true, suggestedMax: 1 } },
      },
    });
  }

  function renderCompareFeatureChart(canvasId, featureChart, leftLabel, rightLabel) {
    createChart(canvasId, {
      type: 'bar',
      data: {
        labels: featureChart.labels,
        datasets: [
          {
            label: leftLabel,
            data: featureChart.left_values,
            backgroundColor: '#0f766e',
          },
          {
            label: rightLabel,
            data: featureChart.right_values,
            backgroundColor: '#c2410c',
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
      },
    });
  }

  function renderCompareEventChart(canvasId, eventChart, leftLabel, rightLabel, title) {
    createChart(canvasId, {
      type: 'bar',
      data: {
        labels: eventChart.event_names,
        datasets: [
          {
            label: leftLabel + ' precision @ 0.5%',
            data: eventChart.left_precision_at_0_5,
            backgroundColor: '#0f766e',
          },
          {
            label: rightLabel + ' precision @ 0.5%',
            data: eventChart.right_precision_at_0_5,
            backgroundColor: '#c2410c',
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: title },
          legend: { position: 'bottom' },
        },
        scales: { y: { beginAtZero: true, suggestedMax: 1 } },
      },
    });
  }

  window.dashboardCharts = {
    renderPrecisionChart: renderPrecisionChart,
    renderFeatureChart: renderFeatureChart,
    renderEventChart: renderEventChart,
    renderComparePrecisionChart: renderComparePrecisionChart,
    renderCompareFeatureChart: renderCompareFeatureChart,
    renderCompareEventChart: renderCompareEventChart,
  };
})();
