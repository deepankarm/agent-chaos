/**
 * agent-chaos dashboard v2 - Narrative-first design
 * Each scenario tells a story.
 */

console.log('🃏 agent-chaos dashboard v2 loaded');

const state = {
    traces: {},
    filter: 'all',
    groupBy: 'none',
    theme: localStorage.getItem('theme') || 'dark',
    expandedCards: new Set(),
};

// ============================================================
// Theme
// ============================================================
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeButton();
}

function toggleTheme() {
    state.theme = state.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('theme', state.theme);
    updateThemeButton();
}

function updateThemeButton() {
    const btn = document.getElementById('themeToggle');
    btn.textContent = state.theme === 'dark' ? '☀️' : '🌙';
    btn.title = state.theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
}

// ============================================================
// Utilities
// ============================================================
function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function formatLatency(ms) {
    if (ms === null || ms === undefined) return '—';
    if (ms < 1000) return Math.round(ms) + 'ms';
    return (ms / 1000).toFixed(2) + 's';
}

function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return '—';
    if (seconds < 1) return (seconds * 1000).toFixed(0) + 'ms';
    return seconds.toFixed(2) + 's';
}

// ============================================================
// Narrative Generation
// ============================================================
function generateNarrative(trace) {
    const report = trace.report || {};
    const meta = report.meta || {};
    const spans = trace.spans || [];
    const faultCount = trace.fault_count || 0;
    const failedCalls = trace.failed_calls || 0;
    const totalCalls = trace.total_calls || spans.length;
    const passed = trace.status === 'success';
    const agentErrored = !!report.error;

    // Extract chaos info from meta
    const chaosType = meta.chaos_type || 'unknown';
    const trigger = meta.trigger || '';
    const kind = meta.kind || 'unknown';

    // Build narrative
    let narrative = '';

    if (kind === 'baseline') {
        narrative = 'Baseline run with no chaos injected.';
    } else if (!faultCount) {
        narrative = 'No faults were injected during this run.';
    } else if (passed && !agentErrored) {
        if (failedCalls > 0) {
            narrative = `Injected ${chaosType} → Agent recovered after ${failedCalls} LLM error(s).`;
        } else {
            narrative = `Injected ${chaosType} → Agent completed successfully.`;
        }
    } else if (passed && agentErrored) {
        narrative = `Injected ${chaosType} → Agent errored as expected (contract passed).`;
    } else {
        narrative = `Injected ${chaosType} → Agent failed to handle the chaos.`;
    }

    if (trigger) {
        narrative += ` Triggered: ${trigger}`;
    }

    return narrative;
}

function getChaosKind(trace) {
    const meta = (trace.report?.meta) || {};
    return meta.kind || 'unknown';
}

function getChaosType(trace) {
    const meta = (trace.report?.meta) || {};
    return meta.chaos_type || 'unknown';
}

function getChaosPills(trace) {
    const meta = (trace.report?.meta) || {};
    const pills = [];
    
    const kind = meta.kind;
    if (kind === 'llm') pills.push({ label: 'LLM', cls: 'llm' });
    else if (kind === 'tool') pills.push({ label: 'Tool', cls: 'tool' });
    else if (kind === 'stream') pills.push({ label: 'Stream', cls: 'stream' });
    else if (kind === 'context') pills.push({ label: 'Context', cls: 'context' });
    else if (kind === 'baseline') pills.push({ label: 'Baseline', cls: '' });
    else if (kind === 'multi') pills.push({ label: 'Multi', cls: 'llm' });
    else if (kind === 'edge') pills.push({ label: 'Edge', cls: 'stream' });

    const chaosType = meta.chaos_type;
    if (chaosType && chaosType !== 'none') {
        pills.push({ label: chaosType.replace(/_/g, ' '), cls: '' });
    }

    return pills;
}

// ============================================================
// Summary Bar
// ============================================================
function updateSummaryBar() {
    const traces = Object.values(state.traces);
    const total = traces.length;
    const passed = traces.filter(t => t.status === 'success').length;
    const failed = total - passed;

    document.getElementById('totalScenarios').textContent = total;
    document.getElementById('passedCount').textContent = passed;
    document.getElementById('failedCount').textContent = failed;

    // Update chaos pills by kind
    const byKind = {};
    traces.forEach(t => {
        const kind = getChaosKind(t);
        if (!byKind[kind]) byKind[kind] = { total: 0, passed: 0 };
        byKind[kind].total++;
        if (t.status === 'success') byKind[kind].passed++;
    });

    const pillsHtml = Object.entries(byKind)
        .filter(([k]) => k !== 'unknown')
        .map(([kind, stats]) => {
            const allPass = stats.passed === stats.total;
            const cls = allPass ? 'all-pass' : 'some-fail';
            return `<span class="chaos-pill ${cls}">${kind}: ${stats.passed}/${stats.total}</span>`;
        })
        .join('');

    document.getElementById('chaosPills').innerHTML = pillsHtml;

    // Show/hide empty state
    const emptyState = document.getElementById('emptyState');
    if (total > 0) {
        emptyState.style.display = 'none';
    } else {
        emptyState.style.display = 'block';
    }
}

// ============================================================
// Filtering & Grouping
// ============================================================
function getFilteredTraces() {
    let traces = Object.values(state.traces);
    
    if (state.filter === 'passed') {
        traces = traces.filter(t => t.status === 'success');
    } else if (state.filter === 'failed') {
        traces = traces.filter(t => t.status !== 'success');
    }

    return traces;
}

function groupTraces(traces) {
    if (state.groupBy === 'none') {
        return [{ name: null, traces }];
    }

    const groups = {};
    traces.forEach(t => {
        let key;
        if (state.groupBy === 'chaos_type') {
            key = getChaosType(t);
        } else if (state.groupBy === 'kind') {
            key = getChaosKind(t);
        } else {
            key = 'unknown';
        }
        if (!groups[key]) groups[key] = [];
        groups[key].push(t);
    });

    return Object.entries(groups).map(([name, traces]) => ({ name, traces }));
}

// ============================================================
// Rendering
// ============================================================
function renderAll() {
    updateSummaryBar();
    const traces = getFilteredTraces();
    const groups = groupTraces(traces);
    const container = document.getElementById('scenarioGroups');
    
    container.innerHTML = groups.map(group => renderGroup(group)).join('');

    // Reattach event listeners
    container.querySelectorAll('.scenario-card').forEach(card => {
        card.addEventListener('click', (e) => {
            // Don't toggle if clicking the view details button
            if (e.target.closest('.view-details-btn')) return;
            const traceId = card.dataset.traceId;
            toggleCard(traceId);
        });
    });

    container.querySelectorAll('.view-details-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const traceId = btn.dataset.traceId;
            openDetailModal(traceId);
        });
    });
}

function renderGroup(group) {
    if (group.name === null) {
        // No grouping
        return group.traces.map(t => renderScenarioCard(t)).join('');
    }

    const passed = group.traces.filter(t => t.status === 'success').length;
    const failed = group.traces.length - passed;

    return `
        <div class="scenario-group">
            <div class="group-header">
                <div class="group-title">${escapeHtml(group.name)}</div>
                <div class="group-stats">
                    <span class="group-stat passed">${passed} passed</span>
                    ${failed > 0 ? `<span class="group-stat failed">${failed} failed</span>` : ''}
                </div>
            </div>
            ${group.traces.map(t => renderScenarioCard(t)).join('')}
        </div>
    `;
}

function renderScenarioCard(trace) {
    const passed = trace.status === 'success';
    const expanded = state.expandedCards.has(trace.trace_id);
    const narrative = generateNarrative(trace);
    const pills = getChaosPills(trace);
    const spans = trace.spans || [];
    const report = trace.report || {};
    const scorecard = report.scorecard || {};

    const pillsHtml = pills.map(p => 
        `<span class="card-pill ${p.cls}">${escapeHtml(p.label)}</span>`
    ).join('');

    const totalCalls = trace.total_calls || spans.length;
    const duration = scorecard.elapsed_s || report.elapsed_s;

    // I/O
    const agentInput = report.agent_input || null;
    const agentOutput = report.agent_output || null;

    // Assertions
    const assertions = report.assertion_results || [];

    return `
        <div class="scenario-card ${passed ? 'passed' : 'failed'} ${expanded ? 'expanded' : ''}" 
             data-trace-id="${trace.trace_id}">
            <div class="card-header">
                <div class="card-status ${passed ? 'passed' : 'failed'}">
                    ${passed ? '✓' : '✗'}
                </div>
                <div class="card-content">
                    <div class="card-title">${escapeHtml(trace.name)}</div>
                    <div class="card-narrative">${escapeHtml(narrative)}</div>
                </div>
                <div class="card-meta">
                    <div class="card-pills">${pillsHtml}</div>
                    <div class="card-stats">
                        <div class="card-stat">
                            <span class="card-stat-value">${totalCalls}</span> calls
                        </div>
                        <div class="card-stat">
                            <span class="card-stat-value">${formatDuration(duration)}</span>
                        </div>
                    </div>
                    <button class="card-toggle" title="Expand">▼</button>
                </div>
            </div>
            <div class="card-body">
                ${renderCardBody(trace, agentInput, agentOutput, assertions)}
            </div>
        </div>
    `;
}

function renderCardBody(trace, agentInput, agentOutput, assertions) {
    const spans = trace.spans || [];
    const scorecard = trace.report?.scorecard || {};

    return `
        <!-- I/O Section -->
        <div class="io-section">
            <div class="io-block">
                <div class="io-header">📥 Agent Input</div>
                <div class="io-content">${agentInput ? escapeHtml(agentInput) : '<span style="color: var(--text-muted)">Not captured</span>'}</div>
            </div>
            <div class="io-block">
                <div class="io-header">📤 Agent Output</div>
                <div class="io-content">${agentOutput ? escapeHtml(agentOutput) : '<span style="color: var(--text-muted)">Not captured</span>'}</div>
            </div>
        </div>

        <!-- Timeline Visual -->
        ${spans.length > 0 ? renderTimelineVisual(spans) : ''}

        <!-- Assertions -->
        ${assertions.length > 0 ? renderAssertions(assertions) : ''}

        <!-- Metrics Row -->
        <div class="metrics-row">
            <div class="metric-box">
                <div class="metric-value">${trace.total_calls || spans.length}</div>
                <div class="metric-label">LLM Calls</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: var(--accent)">${trace.failed_calls || 0}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: var(--warning)">${trace.fault_count || 0}</div>
                <div class="metric-label">Faults</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${scorecard.avg_latency_s ? formatDuration(scorecard.avg_latency_s) : '—'}</div>
                <div class="metric-label">Avg Latency</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${scorecard.avg_ttft_s ? formatDuration(scorecard.avg_ttft_s) : '—'}</div>
                <div class="metric-label">Avg TTFT</div>
            </div>
        </div>

        <button class="view-details-btn" data-trace-id="${trace.trace_id}">
            View Full Details →
        </button>
    `;
}

function renderTimelineVisual(spans) {
    const steps = spans.map((span, i) => {
        const hasError = span.status === 'error';
        const hasChaos = (span.events || []).some(e => e.type === 'fault_injected');
        let dotClass = 'success';
        if (hasError) dotClass = 'error';
        else if (hasChaos) dotClass = 'chaos';
        
        return `
            <div class="timeline-step">
                <div class="timeline-dot ${dotClass}"></div>
                <div class="timeline-label">${i + 1}</div>
            </div>
        `;
    }).join('');

    return `
        <div class="timeline-section">
            <div class="timeline-visual">
                ${steps}
            </div>
        </div>
    `;
}

function renderAssertions(assertions) {
    const items = assertions.map(a => {
        const passed = a.passed;
        return `
            <div class="assertion-item ${passed ? 'passed' : 'failed'}">
                <span class="assertion-icon">${passed ? '✓' : '✗'}</span>
                <span class="assertion-name">${escapeHtml(a.name)}</span>
                ${a.message ? `<span class="assertion-message">${escapeHtml(a.message)}</span>` : ''}
            </div>
        `;
    }).join('');

    return `
        <div class="assertions-section">
            <div class="assertion-list">
                ${items}
            </div>
        </div>
    `;
}

function toggleCard(traceId) {
    if (state.expandedCards.has(traceId)) {
        state.expandedCards.delete(traceId);
    } else {
        state.expandedCards.add(traceId);
    }
    renderAll();
}

// ============================================================
// Detail Modal
// ============================================================
function openDetailModal(traceId) {
    const trace = state.traces[traceId];
    if (!trace) return;

    const modal = document.getElementById('detailModal');
    const title = document.getElementById('modalTitle');
    const body = document.getElementById('modalBody');

    title.textContent = trace.name;
    body.innerHTML = renderModalBody(trace);

    modal.classList.remove('hidden');
}

function closeDetailModal() {
    document.getElementById('detailModal').classList.add('hidden');
}

function renderModalBody(trace) {
    const spans = trace.spans || [];
    const report = trace.report || {};
    const scorecard = report.scorecard || {};

    return `
        <!-- Summary -->
        <div class="modal-section">
            <div class="modal-section-title">Summary</div>
            <div class="metrics-row">
                <div class="metric-box">
                    <div class="metric-value">${trace.status === 'success' ? '✓ Pass' : '✗ Fail'}</div>
                    <div class="metric-label">Result</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value">${trace.total_calls || spans.length}</div>
                    <div class="metric-label">LLM Calls</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: var(--accent)">${trace.failed_calls || 0}</div>
                    <div class="metric-label">Errors</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color: var(--warning)">${trace.fault_count || 0}</div>
                    <div class="metric-label">Faults Injected</div>
                </div>
            </div>
        </div>

        <!-- I/O -->
        <div class="modal-section">
            <div class="modal-section-title">Agent I/O</div>
            <div class="io-section">
                <div class="io-block">
                    <div class="io-header">📥 Input</div>
                    <div class="io-content">${report.agent_input ? escapeHtml(report.agent_input) : 'Not captured'}</div>
                </div>
                <div class="io-block">
                    <div class="io-header">📤 Output</div>
                    <div class="io-content">${report.agent_output ? escapeHtml(report.agent_output) : 'Not captured'}</div>
                </div>
            </div>
        </div>

        <!-- Error (if any) -->
        ${report.error ? `
        <div class="modal-section">
            <div class="modal-section-title">Error</div>
            <div class="io-block">
                <div class="io-content" style="color: var(--accent)">${escapeHtml(report.error)}</div>
            </div>
        </div>
        ` : ''}

        <!-- Assertions -->
        ${(report.assertion_results || []).length > 0 ? `
        <div class="modal-section">
            <div class="modal-section-title">Assertions</div>
            ${renderAssertions(report.assertion_results)}
        </div>
        ` : ''}

        <!-- Waterfall -->
        ${spans.length > 0 ? `
        <div class="modal-section">
            <div class="modal-section-title">LLM Calls Waterfall</div>
            ${renderWaterfall(spans)}
        </div>
        ` : ''}

        <!-- Raw Scorecard -->
        <div class="modal-section">
            <div class="modal-section-title">Raw Scorecard</div>
            <div class="io-block">
                <div class="io-content">${escapeHtml(JSON.stringify(scorecard, null, 2))}</div>
            </div>
        </div>
    `;
}

function renderWaterfall(spans) {
    const maxLatency = Math.max(1, ...spans.map(s => s.latency_ms || 0));

    const rows = spans.map(span => {
        const barWidth = span.latency_ms 
            ? Math.max(5, (span.latency_ms / maxLatency) * 100) 
            : 5;
        const shortId = span.span_id.split('_').slice(0, 2).join('_');
        
        return `
            <div class="waterfall-row">
                <div class="span-id">${shortId}</div>
                <div class="span-provider">${span.provider || 'unknown'}</div>
                <div class="span-bar-container">
                    <div class="span-bar ${span.status}" style="width: ${barWidth}%"></div>
                </div>
                <div class="span-latency ${span.status}">${formatLatency(span.latency_ms)}</div>
            </div>
        `;
    }).join('');

    return `<div class="waterfall">${rows}</div>`;
}

// ============================================================
// Event Handlers
// ============================================================
function setupEventHandlers() {
    // Filter change
    document.getElementById('filterStatus').addEventListener('change', (e) => {
        state.filter = e.target.value;
        renderAll();
    });

    // Group by change
    document.getElementById('groupBy').addEventListener('change', (e) => {
        state.groupBy = e.target.value;
        renderAll();
    });

    // Modal close
    document.getElementById('modalClose').addEventListener('click', closeDetailModal);
    document.querySelector('.modal-backdrop').addEventListener('click', closeDetailModal);
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDetailModal();
    });
}

// ============================================================
// Data Loading
// ============================================================
function loadTraces() {
    fetch('/api/traces?include_artifacts=true')
        .then(r => r.json())
        .then(traces => {
            traces.forEach(trace => {
                state.traces[trace.trace_id] = trace;
            });
            renderAll();
        })
        .catch(err => console.error('Failed to load traces:', err));
}

// ============================================================
// WebSocket Connection
// ============================================================
function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    ws.onopen = () => {
        document.getElementById('statusBadge').className = 'status-badge connected';
        document.getElementById('statusText').textContent = 'Connected';
    };
    
    ws.onclose = () => {
        document.getElementById('statusBadge').className = 'status-badge disconnected';
        document.getElementById('statusText').textContent = 'Disconnected';
        setTimeout(connect, 2000);
    };
    
    ws.onerror = () => {
        document.getElementById('statusBadge').className = 'status-badge disconnected';
        document.getElementById('statusText').textContent = 'Error';
    };
    
    ws.onmessage = (msg) => {
        try {
            const event = JSON.parse(msg.data);
            handleLiveEvent(event);
        } catch (e) {
            console.error('Failed to parse event:', e);
        }
    };
}

function handleLiveEvent(event) {
    // For live events, we update the trace in-place
    // Similar logic to v1 but simplified
    switch (event.type) {
        case 'trace_start':
            state.traces[event.trace_id] = {
                trace_id: event.trace_id,
                name: event.trace_name,
                start_time: event.timestamp,
                status: 'running',
                total_calls: 0,
                failed_calls: 0,
                fault_count: 0,
                spans: [],
            };
            renderAll();
            break;
            
        case 'trace_end':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                trace.end_time = event.timestamp;
                trace.status = trace.spans.some(s => s.status === 'error') ? 'error' : 'success';
                if (event.data) {
                    trace.total_calls = event.data.total_calls;
                    trace.fault_count = event.data.fault_count || event.data.chaos_count;
                }
                renderAll();
            }
            break;
            
        case 'span_start':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                trace.spans.push({
                    span_id: event.span_id,
                    provider: event.provider,
                    status: 'running',
                    latency_ms: null,
                    events: [],
                });
                trace.total_calls = trace.spans.length;
                renderAll();
            }
            break;
            
        case 'span_end':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                const span = trace.spans.find(s => s.span_id === event.span_id);
                if (span) {
                    span.status = event.data?.success ? 'success' : 'error';
                    span.latency_ms = event.data?.latency_ms;
                    span.error = event.data?.error || '';
                    if (!event.data?.success) {
                        trace.failed_calls = (trace.failed_calls || 0) + 1;
                    }
                }
                renderAll();
            }
            break;
            
        case 'fault_injected':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                trace.fault_count = (trace.fault_count || 0) + 1;
                const span = trace.spans.find(s => s.span_id === event.span_id);
                if (span) {
                    span.events = span.events || [];
                    span.events.push(event);
                }
                renderAll();
            }
            break;
    }
}

// ============================================================
// Init
// ============================================================
function init() {
    initTheme();
    setupEventHandlers();
    loadTraces();
    connect();

    // Poll for artifact traces every 3 seconds
    setInterval(loadTraces, 3000);
}

document.addEventListener('DOMContentLoaded', init);

