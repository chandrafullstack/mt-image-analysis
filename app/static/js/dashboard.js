// Dashboard JavaScript — Plotly charts with hover image popup + tabs

let chartData = [];
let neuronData = [];
let currentImageFilter = null;  // when set, renderChart shows only that source_file's mitos

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
    renderCurrentView();  // start in image-overview mode by default
    renderNeuronChart();
    renderCrossTables();
    initTabs();
    initTableToggle();
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

// --- Collapsible data table (renders lazily on first expand) ---
let tableRendered = false;
function initTableToggle() {
    const btn = document.getElementById('data-table-toggle');
    const wrap = document.getElementById('data-table-wrapper');
    if (!btn || !wrap) return;
    btn.addEventListener('click', () => {
        const isHidden = wrap.hasAttribute('hidden');
        if (isHidden) {
            if (!tableRendered) { renderTable(); tableRendered = true; }
            wrap.removeAttribute('hidden');
            btn.setAttribute('aria-expanded', 'true');
            btn.textContent = '▾ Hide data table';
        } else {
            wrap.setAttribute('hidden', '');
            btn.setAttribute('aria-expanded', 'false');
            btn.textContent = '▸ Show data table';
        }
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

    const sourceData = currentImageFilter
        ? chartData.filter(d => d.source_file === currentImageFilter)
        : chartData;

    for (const d of sourceData) {
        const key = d[colorField] || 'UNKNOWN';
        if (!groups[key]) groups[key] = [];
        groups[key].push(d);
    }

    const traces = Object.entries(groups).map(([name, data]) => ({
        x: data.map(d => d[xField]),
        y: data.map(d => d.g_ratio),
        customdata: data,
        mode: 'markers',
        type: 'scattergl',
        name: name,
        marker: { color: colorMap[name] || '#888', size: 7, opacity: 0.7,
                  line: { width: 0.5, color: '#fff' } },
        hoverinfo: 'text',
        hovertext: data.map(d =>
            `ID: ${d.instance_id}<br>G-ratio: ${d.g_ratio}<br>` +
            `AR: ${d.aspect_ratio}<br>Area: ${d.area_um2} µm²<br>` +
            `Health: ${d.classification}<br>Shape: ${d.shape_category}<br>` +
            `State: ${d.fission_fusion}<br>Myelin: ${d.myelin_context}`
        ),
    }));

    const titleText = currentImageFilter
        ? `Mitochondria in <i>${currentImageFilter}</i> (${sourceData.length} mitos) · drag to zoom`
        : `Mitochondria — ${sourceData.length} total · hover for EM image · drag to zoom`;
    const layout = {
        title: { text: titleText, font: { color: '#eee', size: 14 } },
        xaxis: { title: xField.replace(/_/g, ' '), color: '#aaa', gridcolor: '#2a3a5e', autorange: true },
        yaxis: { title: 'Mito G-ratio', color: '#aaa', gridcolor: '#2a3a5e', range: [0.3, 1.05], fixedrange: false },
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: { color: '#eee' },
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(22,33,62,0.8)' },
        shapes: [{ type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 0.6, y1: 0.8,
                   fillcolor: 'rgba(0,212,170,0.06)', line: { width: 0 } }],
        dragmode: 'zoom',
        hovermode: 'closest',
    };

    const config = {
        responsive: true,
        scrollZoom: true,
        displayModeBar: true,
        displaylogo: false,
        modeBarButtonsToAdd: ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'],
        toImageButtonOptions: { format: 'png', filename: 'mito_gratio', scale: 2 },
    };

    Plotly.newPlot('gratio-chart', traces, layout, config);

    // Inspector panel (side-by-side, persistent)
    const chartEl = document.getElementById('gratio-chart');
    const inspectorImg = document.getElementById('inspector-img');
    const inspectorCrop = document.getElementById('inspector-crop');
    const inspectorCropPh = document.getElementById('inspector-crop-placeholder');
    const inspectorCropBox = document.getElementById('inspector-cropbox');
    const inspectorCropStage = document.getElementById('inspector-crop-stage');
    const inspectorPh = document.getElementById('inspector-placeholder');
    const inspectorMeta = document.getElementById('inspector-meta');

    let lastInspectorId = null;
    let lastSourceFile = null;
    let lastClickedId = null;

    // --- Zoom / pan state for crop ---
    let zoom = 1, panX = 0, panY = 0;
    function applyTransform() {
        inspectorCropStage.style.transform =
            `translate(${panX}px, ${panY}px) scale(${zoom})`;
    }
    function resetZoom() { zoom = 1; panX = 0; panY = 0; applyTransform(); }

    if (inspectorCropBox) {
        inspectorCropBox.addEventListener('wheel', (e) => {
            if (inspectorCrop.style.display === 'none') return;
            e.preventDefault();
            const rect = inspectorCropBox.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            const newZoom = Math.max(1, Math.min(20, zoom * factor));
            // Zoom toward cursor: keep the point under the cursor stable
            panX = cx - (cx - panX) * (newZoom / zoom);
            panY = cy - (cy - panY) * (newZoom / zoom);
            zoom = newZoom;
            if (zoom === 1) { panX = 0; panY = 0; }
            applyTransform();
        }, { passive: false });

        let dragging = false, dragStartX = 0, dragStartY = 0, startPanX = 0, startPanY = 0;
        inspectorCropBox.addEventListener('mousedown', (e) => {
            if (inspectorCrop.style.display === 'none') return;
            dragging = true;
            dragStartX = e.clientX; dragStartY = e.clientY;
            startPanX = panX; startPanY = panY;
        });
        window.addEventListener('mousemove', (e) => {
            if (!dragging) return;
            panX = startPanX + (e.clientX - dragStartX);
            panY = startPanY + (e.clientY - dragStartY);
            applyTransform();
        });
        window.addEventListener('mouseup', () => { dragging = false; });
        inspectorCropBox.addEventListener('dblclick', resetZoom);
    }

    function updateMeta(d) {
        const rows = [
            ['ID', d.instance_id],
            ['Source image', d.source_file],
            ['G-ratio', d.g_ratio],
            ['Health', d.classification],
            ['Shape', d.shape_category],
            ['State', d.fission_fusion],
            ['Myelin context', d.myelin_context],
            ['Aspect ratio', d.aspect_ratio],
            ['Area (µm²)', d.area_um2],
            ['Form factor', d.form_factor],
        ];
        inspectorMeta.innerHTML = rows.map(([k, v]) =>
            `<dt>${k}</dt><dd>${v ?? '—'}</dd>`).join('');
        lastInspectorId = d.instance_id;
    }

    function loadCrop(d) {
        if (!inspectorCrop) return;
        resetZoom();
        if (d.crop_image) {
            inspectorCrop.src = d.crop_image;
            inspectorCrop.style.display = 'block';
            if (inspectorCropPh) inspectorCropPh.style.display = 'none';
        } else {
            inspectorCrop.style.display = 'none';
            if (inspectorCropPh) {
                inspectorCropPh.style.display = 'flex';
                inspectorCropPh.textContent = 'No crop available.';
            }
        }
    }

    function loadSourceWithBbox(d) {
        if (d.instance_id == null) return;
        if (d.instance_id === lastClickedId) return;
        lastClickedId = d.instance_id;

        inspectorPh.style.display = 'block';
        inspectorPh.textContent = lastSourceFile === d.source_file
            ? 'Updating box…' : 'Loading source image…';
        inspectorImg.style.display = 'none';

        const ctxUrl = `/api/instance/${d.instance_id}/context`;
        const probe = new Image();
        probe.onload = () => {
            if (lastClickedId !== d.instance_id) return;
            inspectorImg.src = probe.src;
            inspectorImg.style.display = 'block';
            inspectorPh.style.display = 'none';
            lastSourceFile = d.source_file;
        };
        probe.onerror = () => {
            if (lastClickedId !== d.instance_id) return;
            inspectorImg.style.display = 'none';
            inspectorPh.style.display = 'block';
            inspectorPh.textContent = 'No source image available.';
        };
        probe.src = ctxUrl;
    }

    chartEl.on('plotly_hover', function (event) {
        const d = event.points[0]?.customdata;
        if (!d) return;
        // Hover = metadata only. Images don't move.
        if (d.instance_id !== lastInspectorId) updateMeta(d);
    });
    chartEl.on('plotly_click', function (event) {
        const d = event.points[0]?.customdata;
        if (!d) return;
        updateMeta(d);
        loadCrop(d);
        loadSourceWithBbox(d);
    });
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

// --- Image-overview chart: one dot per source image ---
function renderImageChart() {
    // Aggregate per source_file
    const byImage = {};
    for (const d of chartData) {
        const k = d.source_file || 'unknown';
        if (!byImage[k]) {
            byImage[k] = { source_file: k, mitos: [], resolution_group: d.resolution_group };
        }
        byImage[k].mitos.push(d);
    }

    const images = Object.values(byImage).map(img => {
        const n = img.mitos.length;
        const gratios = img.mitos.map(m => m.g_ratio).filter(g => g > 0);
        const medianG = gratios.length ? gratios.sort((a, b) => a - b)[Math.floor(gratios.length / 2)] : 0;
        const unhealthy = img.mitos.filter(m => m.classification === 'UNHEALTHY').length;
        const pctUnhealthy = n ? (unhealthy / n) * 100 : 0;
        return {
            source_file: img.source_file,
            resolution_group: img.resolution_group,
            n: n,
            median_gratio: round4(medianG),
            pct_unhealthy: Math.round(pctUnhealthy),
        };
    }).sort((a, b) => a.source_file.localeCompare(b.source_file));

    // Color by resolution group
    const colorByRes = { '200nm': '#42a5f5', '400nm': '#ffa726', '800nm': '#ab47bc', 'unknown': '#888' };
    const groups = {};
    for (const img of images) {
        const k = img.resolution_group || 'unknown';
        if (!groups[k]) groups[k] = [];
        groups[k].push(img);
    }

    const traces = Object.entries(groups).map(([name, data]) => ({
        x: data.map(d => d.source_file),
        y: data.map(d => d.median_gratio),
        customdata: data,
        mode: 'markers',
        type: 'scattergl',
        name: name,
        marker: {
            color: colorByRes[name] || '#888',
            size: data.map(d => Math.min(60, 8 + Math.sqrt(d.n))),
            opacity: 0.75,
            line: { width: 1, color: '#fff' },
        },
        hoverinfo: 'text',
        hovertext: data.map(d =>
            `${d.source_file}<br>Resolution: ${d.resolution_group}<br>` +
            `Mitos: ${d.n}<br>Median G-ratio: ${d.median_gratio}<br>` +
            `% Unhealthy: ${d.pct_unhealthy}%<br><i>Click to drill in</i>`),
    }));

    const layout = {
        title: { text: `Per-image overview — ${images.length} images · dot size = mito count · click to drill in`, font: { color: '#eee', size: 14 } },
        xaxis: { title: 'Source image', color: '#aaa', gridcolor: '#2a3a5e', tickangle: -45, automargin: true },
        yaxis: { title: 'Median G-ratio (per image)', color: '#aaa', gridcolor: '#2a3a5e', range: [0.3, 1.05] },
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: { color: '#eee' },
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(22,33,62,0.8)' },
        hovermode: 'closest',
        dragmode: 'zoom',
    };

    Plotly.newPlot('gratio-chart', traces, layout, {
        responsive: true, scrollZoom: true, displaylogo: false,
    });

    // Clear inspector
    const inspectorPh = document.getElementById('inspector-placeholder');
    const inspectorImg = document.getElementById('inspector-img');
    const inspectorMeta = document.getElementById('inspector-meta');
    inspectorImg.style.display = 'none';
    inspectorPh.style.display = 'block';
    inspectorPh.textContent = 'Click an image dot to drill in and see its mitochondria.';
    inspectorMeta.innerHTML = '';

    document.getElementById('gratio-chart').on('plotly_click', function (event) {
        const d = event.points[0]?.customdata;
        if (d && d.source_file) {
            currentImageFilter = d.source_file;
            document.getElementById('view-select').value = 'mito';
            document.getElementById('back-btn').style.display = 'inline-block';
            renderChart(
                document.getElementById('x-select').value,
                document.getElementById('color-select').value,
            );
        }
    });

    document.getElementById('gratio-chart').on('plotly_hover', function (event) {
        const d = event.points[0]?.customdata;
        if (!d) return;
        // Load full source image into inspector
        const url = `/api/image-preview?source_file=${encodeURIComponent(d.source_file)}`;
        inspectorImg.onerror = () => {
            inspectorImg.style.display = 'none';
            inspectorPh.style.display = 'block';
            inspectorPh.textContent = 'Could not load preview for this image.';
        };
        inspectorImg.onload = () => {
            inspectorImg.style.display = 'block';
            inspectorPh.style.display = 'none';
        };
        inspectorImg.src = url;
        const rows = [
            ['Image', d.source_file],
            ['Resolution', d.resolution_group],
            ['Mito count', d.n],
            ['Median G-ratio', d.median_gratio],
            ['% Unhealthy', d.pct_unhealthy + '%'],
            ['', '(click dot to drill in)'],
        ];
        inspectorMeta.innerHTML = rows.map(([k, v]) =>
            `<dt>${k}</dt><dd>${v ?? '—'}</dd>`).join('');
    });
}

function round4(x) { return Math.round(x * 10000) / 10000; }

function renderCurrentView() {
    const mode = document.getElementById('view-select').value;
    const backBtn = document.getElementById('back-btn');
    if (mode === 'image') {
        currentImageFilter = null;
        backBtn.style.display = 'none';
        renderImageChart();
    } else {
        backBtn.style.display = currentImageFilter ? 'inline-block' : 'none';
        renderChart(
            document.getElementById('x-select').value,
            document.getElementById('color-select').value,
        );
    }
}

// Controls
document.getElementById('x-select').addEventListener('change', (e) => {
    if (document.getElementById('view-select').value === 'mito') {
        renderChart(e.target.value, document.getElementById('color-select').value);
    }
});
document.getElementById('color-select').addEventListener('change', (e) => {
    if (document.getElementById('view-select').value === 'mito') {
        renderChart(document.getElementById('x-select').value, e.target.value);
    }
});
document.getElementById('view-select').addEventListener('change', () => {
    currentImageFilter = null;
    renderCurrentView();
});
document.getElementById('back-btn').addEventListener('click', () => {
    currentImageFilter = null;
    document.getElementById('view-select').value = 'image';
    renderCurrentView();
});

// --- Research Assistant (chat) ---
const chatLog = document.getElementById('chat-log');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');
const keyStatusText = document.getElementById('key-status-text');
const keyForm = document.getElementById('key-form');
const keyInput = document.getElementById('key-input');

async function refreshKeyStatus() {
    if (!keyStatusText) return;
    try {
        const res = await fetch('/api/chat/key-status');
        const data = await res.json();
        if (data.configured) {
            keyStatusText.textContent = `✓ API key configured (${data.source}). You're ready to chat.`;
            keyStatusText.style.color = '#00d4aa';
            if (keyForm) keyForm.style.display = 'none';
            if (chatSend) chatSend.disabled = false;
        } else {
            keyStatusText.textContent = 'No ANTHROPIC_API_KEY set. Paste one below to enable the agent (session only — not saved to disk).';
            keyStatusText.style.color = '#ffa726';
            if (keyForm) keyForm.style.display = 'flex';
            if (chatSend) chatSend.disabled = true;
        }
    } catch (err) {
        keyStatusText.textContent = 'Could not check key status: ' + err.message;
    }
}

if (keyForm) {
    keyForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const key = keyInput.value.trim();
        if (!key) return;
        try {
            const res = await fetch('/api/chat/set-key', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ key }),
            });
            const data = await res.json();
            if (data.ok) {
                keyInput.value = '';
                await refreshKeyStatus();
            } else {
                keyStatusText.textContent = 'Error: ' + (data.error || 'failed to set key');
                keyStatusText.style.color = '#ff6b6b';
            }
        } catch (err) {
            keyStatusText.textContent = 'Network error: ' + err.message;
        }
    });
    refreshKeyStatus();
}

function addBubble(text, cls) {
    const div = document.createElement('div');
    div.className = 'chat-bubble ' + cls;
    div.textContent = text;
    chatLog.appendChild(div);
    chatLog.scrollTop = chatLog.scrollHeight;
    return div;
}

function renderFollowups(followups) {
    const bar = document.getElementById('followups-bar');
    if (!bar) return;
    bar.innerHTML = '';
    if (!followups || !followups.length) return;
    const label = document.createElement('span');
    label.className = 'chip-label';
    label.textContent = 'Follow up:';
    bar.appendChild(label);
    followups.forEach(q => {
        const btn = document.createElement('button');
        btn.className = 'chip followup';
        btn.textContent = q;
        btn.addEventListener('click', () => askQuestion(q));
        bar.appendChild(btn);
    });
}

async function askQuestion(q) {
    if (!q || !q.trim()) return;
    addBubble(q, 'user');
    chatInput.value = '';
    chatSend.disabled = true;
    renderFollowups([]);  // clear stale chips
    const thinking = addBubble('Thinking…', 'assistant');
    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q }),
        });
        const data = await res.json();
        thinking.textContent = data.answer || '(empty)';
        if (data.tool_calls && data.tool_calls.length) {
            const summary = 'Tools used: ' + data.tool_calls.map(t => t.name).join(', ')
                + ` (${data.iterations} iter${data.iterations > 1 ? 's' : ''})`;
            addBubble(summary, 'tool-trace');
        }
        renderFollowups(data.followups || []);
    } catch (err) {
        thinking.textContent = 'Error: ' + err.message;
    } finally {
        chatSend.disabled = false;
        chatInput.focus();
    }
}

if (chatForm) {
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        askQuestion(chatInput.value.trim());
    });
}

// FAQ quick-start chips
document.querySelectorAll('.faq-chips .chip').forEach(btn => {
    btn.addEventListener('click', () => {
        const q = btn.getAttribute('data-q');
        if (q) askQuestion(q);
    });
});

// Init
loadData();
