// Dashboard JavaScript — Plotly charts with hover image popup + tabs

let chartData = [];
let neuronData = [];

async function loadData() {
    const [dataRes, summaryRes, neuronRes] = await Promise.all([
        fetch('/api/gratio-data'),
        fetch('/api/summary'),
        fetch('/api/neuron-gratio'),
    ]);
    chartData = await dataRes.json();
    const summary = await summaryRes.json();
    neuronData = await neuronRes.json();

    renderSummary(summary);
    renderChart('instance_id', 'classification');
    renderNeuronChart();
    renderCrossTables();
    renderTable();
    initTabs();
}

function renderSummary(s) {
    document.getElementById('summary-stats').innerHTML = `
        <span>Mitos: <strong>${s.total_mitochondria}</strong></span>
        <span>Healthy: <strong style="color:#00d4aa">${s.healthy_count}</strong></span>
        <span>Unhealthy: <strong style="color:#ff6b6b">${s.unhealthy_count}</strong></span>
        <span>Fission: <strong style="color:#ffa726">${s.fission_count}</strong></span>
        <span>Fusion: <strong style="color:#ab47bc">${s.fusion_count}</strong></span>
        <span>Neurons: <strong>${s.neuron_count}</strong></span>
        <span>Neuron G-ratio: <strong>${s.mean_neuron_gratio ?? '—'}</strong></span>
    `;
}

// --- Tab management ---
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(btn.dataset.tab).classList.add('active');
        });
    });
}

// --- Color schemes for different groupings ---
const COLOR_MAPS = {
    classification: { HEALTHY: '#00d4aa', UNHEALTHY: '#ff6b6b', UNKNOWN: '#888' },
    shape_category: { ELONGATED: '#42a5f5', CIRCULAR: '#ffa726', OTHER: '#888' },
    fission_fusion: { NORMAL: '#00d4aa', FISSION: '#ffa726', FUSION: '#ab47bc' },
    myelin_context: { WELL_MYELINATED: '#00d4aa', POORLY_MYELINATED: '#ff6b6b', UNASSIGNED: '#555' },
};

function renderChart(xField, colorField) {
    const colorMap = COLOR_MAPS[colorField] || COLOR_MAPS.classification;
    const groups = {};

    for (const d of chartData) {
        const key = d[colorField] || 'UNKNOWN';
        if (!groups[key]) groups[key] = [];
        groups[key].push(d);
    }

    const traces = Object.entries(groups).map(([name, data]) => ({
        x: data.map(d => d[xField]),
        y: data.map(d => d.g_ratio),
        customdata: data,
        mode: 'markers',
        type: 'scatter',
        name: name,
        marker: { color: colorMap[name] || '#888', size: 9, opacity: 0.8,
                  line: { width: 1, color: '#fff' } },
        hoverinfo: 'text',
        hovertext: data.map(d =>
            `ID: ${d.instance_id}<br>G-ratio: ${d.g_ratio}<br>` +
            `AR: ${d.aspect_ratio}<br>Area: ${d.area_um2} µm²<br>` +
            `Health: ${d.classification}<br>Shape: ${d.shape_category}<br>` +
            `State: ${d.fission_fusion}<br>Myelin: ${d.myelin_context}`
        ),
    }));

    const layout = {
        title: { text: 'Mitochondria — Hover for EM Image', font: { color: '#eee', size: 16 } },
        xaxis: { title: xField.replace(/_/g, ' '), color: '#aaa', gridcolor: '#2a3a5e' },
        yaxis: { title: 'Mito G-ratio', color: '#aaa', gridcolor: '#2a3a5e', range: [0.3, 1.05] },
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: { color: '#eee' },
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(22,33,62,0.8)' },
        shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0.6, y1: 0.8,
                   fillcolor: 'rgba(0,212,170,0.06)', line: { width: 0 } }],
    };

    Plotly.newPlot('gratio-chart', traces, layout, { responsive: true });

    // Hover popup
    const chartEl = document.getElementById('gratio-chart');
    const popup = document.getElementById('image-popup');
    const popupImg = document.getElementById('popup-img');
    const popupInfo = document.getElementById('popup-info');

    chartEl.on('plotly_hover', function (event) {
        const point = event.points[0];
        const d = point.customdata;
        if (d && d.crop_image) {
            popupImg.src = d.crop_image;
            popupInfo.textContent = `#${d.instance_id} | G=${d.g_ratio} | ${d.classification} | ${d.shape_category}`;
            const xPx = point.xaxis.l2p(point.x) + point.xaxis._offset;
            const yPx = point.yaxis.l2p(point.y) + point.yaxis._offset;
            let left = xPx + 24, top = yPx - 110;
            if (left + 220 > chartEl.clientWidth) left = xPx - 230;
            if (top < 0) top = yPx + 20;
            popup.style.left = left + 'px';
            popup.style.top = top + 'px';
            popup.style.display = 'block';
        }
    });
    chartEl.on('plotly_unhover', () => { popup.style.display = 'none'; });
}

// --- Neuron G-ratio chart ---
function renderNeuronChart() {
    if (neuronData.length === 0) {
        document.getElementById('neuron-gratio-chart').innerHTML =
            '<p style="color:#888;text-align:center;padding:40px">No neuron G-ratio data available yet. Run myelin segmentation first.</p>';
        return;
    }

    const well = neuronData.filter(d => d.myelin_health === 'WELL_MYELINATED');
    const poor = neuronData.filter(d => d.myelin_health !== 'WELL_MYELINATED');

    const traces = [
        { x: well.map(d => d.neuron_id), y: well.map(d => d.g_ratio),
          type: 'bar', name: 'Well Myelinated', marker: { color: '#00d4aa' } },
        { x: poor.map(d => d.neuron_id), y: poor.map(d => d.g_ratio),
          type: 'bar', name: 'Poorly Myelinated', marker: { color: '#ff6b6b' } },
    ];

    const layout = {
        title: { text: 'Per-Neuron G-Ratio (Axon/Fibre)', font: { color: '#eee', size: 16 } },
        xaxis: { title: 'Neuron ID', color: '#aaa', gridcolor: '#2a3a5e' },
        yaxis: { title: 'G-ratio (d_axon / d_fibre)', color: '#aaa', gridcolor: '#2a3a5e',
                 range: [0, 1.2] },
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e', font: { color: '#eee' },
        legend: { x: 0.01, y: 0.99 },
        shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0.6, y1: 0.8,
                   fillcolor: 'rgba(0,212,170,0.08)', line: { width: 0 } }],
    };

    Plotly.newPlot('neuron-gratio-chart', traces, layout, { responsive: true });
}

// --- Cross-tabulation tables ---
function renderCrossTables() {
    // Health × Myelin
    const healthMyelin = crossTab('classification', 'myelin_context',
        ['HEALTHY', 'UNHEALTHY'], ['WELL_MYELINATED', 'POORLY_MYELINATED', 'UNASSIGNED']);
    document.getElementById('health-myelin-table').innerHTML =
        '<div class="cross-table"><h3>Health × Myelin Quality</h3>' +
        buildTable(['', 'Well Myelinated', 'Poorly Myelinated', 'Unassigned'],
            [['Healthy', ...healthMyelin['HEALTHY']],
             ['Unhealthy', ...healthMyelin['UNHEALTHY']]]) + '</div>';

    // Shape × Myelin
    const shapeMyelin = crossTab('shape_category', 'myelin_context',
        ['ELONGATED', 'CIRCULAR', 'OTHER'], ['WELL_MYELINATED', 'POORLY_MYELINATED', 'UNASSIGNED']);
    document.getElementById('shape-myelin-table').innerHTML =
        '<div class="cross-table"><h3>Shape × Myelin Quality</h3>' +
        buildTable(['', 'Well Myelinated', 'Poorly Myelinated', 'Unassigned'],
            [['Elongated', ...shapeMyelin['ELONGATED']],
             ['Circular', ...shapeMyelin['CIRCULAR']],
             ['Other', ...shapeMyelin['OTHER']]]) + '</div>';

    // Fission/Fusion × Myelin
    const ffMyelin = crossTab('fission_fusion', 'myelin_context',
        ['NORMAL', 'FISSION', 'FUSION'], ['WELL_MYELINATED', 'POORLY_MYELINATED', 'UNASSIGNED']);
    document.getElementById('fission-myelin-table').innerHTML =
        '<div class="cross-table"><h3>Fission/Fusion × Myelin Quality</h3>' +
        buildTable(['', 'Well Myelinated', 'Poorly Myelinated', 'Unassigned'],
            [['Normal', ...ffMyelin['NORMAL']],
             ['Fission', ...ffMyelin['FISSION']],
             ['Fusion', ...ffMyelin['FUSION']]]) + '</div>';
}

function crossTab(rowField, colField, rowValues, colValues) {
    const result = {};
    for (const rv of rowValues) {
        result[rv] = colValues.map(cv =>
            chartData.filter(d => d[rowField] === rv && d[colField] === cv).length
        );
    }
    return result;
}

function buildTable(headers, rows) {
    let html = '<table><thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    rows.forEach(row => {
        html += '<tr>';
        row.forEach((cell, i) => html += i === 0 ? `<td><strong>${cell}</strong></td>` : `<td>${cell}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table>';
    return html;
}

function renderTable() {
    if (chartData.length === 0) {
        document.getElementById('data-table-container').innerHTML = '<p style="color:#888">No data.</p>';
        return;
    }
    let html = `<table><thead><tr>
        <th>ID</th><th>G-ratio</th><th>AR</th><th>Area</th>
        <th>Shape</th><th>Health</th><th>Fission/Fusion</th><th>Myelin</th>
    </tr></thead><tbody>`;
    for (const d of chartData) {
        const cls = d.classification === 'HEALTHY' ? 'healthy' : 'unhealthy';
        html += `<tr>
            <td>${d.instance_id}</td><td>${d.g_ratio}</td><td>${d.aspect_ratio}</td>
            <td>${d.area_um2}</td><td>${d.shape_category}</td>
            <td class="${cls}">${d.classification}</td>
            <td>${d.fission_fusion}</td><td>${d.myelin_context}</td>
        </tr>`;
    }
    html += '</tbody></table>';
    document.getElementById('data-table-container').innerHTML = html;
}

// Controls
document.getElementById('x-select').addEventListener('change', (e) => {
    renderChart(e.target.value, document.getElementById('color-select').value);
});
document.getElementById('color-select').addEventListener('change', (e) => {
    renderChart(document.getElementById('x-select').value, e.target.value);
});

// Init
loadData();
