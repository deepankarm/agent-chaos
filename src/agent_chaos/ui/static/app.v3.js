/**
 * agent-chaos dashboard v3.3
 * Narrative-first, compact UI with proper flame graph
 */

console.log('🃏 agent-chaos dashboard v3.3 loaded');

// ============================================================
// State
// ============================================================
const state = {
    traces: {},
    tracesHash: '',
    theme: localStorage.getItem('theme') || 'dark',
    filter: 'all',
    selectedTraceId: null,
};

// ============================================================
// Utilities
// ============================================================
function escapeHtml(str) {
    if (str == null) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function formatTime(isoString) {
    if (!isoString) return '—';
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { 
        hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' 
    });
}

function formatDuration(seconds) {
    if (seconds == null) return '—';
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function formatLatency(ms) {
    if (ms == null) return '—';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

// Hash to detect real changes
function computeTracesHash() {
    const keys = Object.keys(state.traces).sort();
    const parts = keys.map(k => {
        const t = state.traces[k];
        const spanCount = (t.spans || []).length;
        const eventCount = (t.spans || []).reduce((sum, s) => sum + (s.events || []).length, 0);
        return `${k}:${t.status}:${t.total_calls}:${t.fault_count}:${spanCount}:${eventCount}`;
    });
    return parts.join('|');
}

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
// Data Extraction
// ============================================================
function extractFaults(trace) {
    const faults = [];
    (trace.spans || []).forEach(s => {
        (s.events || []).forEach(e => {
            if (e.type === 'fault_injected') {
                faults.push({
                    type: e.data?.fault_type || 'unknown',
                    spanId: s.span_id,
                    timestamp: e.timestamp,
                });
            }
        });
    });
    return faults;
}

// Only use tool_end events, dedupe by tool_use_id
function extractTools(trace) {
    const toolsMap = new Map();
    (trace.spans || []).forEach(s => {
        (s.events || []).forEach(e => {
            if (e.type === 'tool_end') {
                const id = e.data?.tool_use_id || e.data?.tool_name || `${s.span_id}_${e.timestamp}`;
                if (!toolsMap.has(id)) {
                    toolsMap.set(id, {
                        id,
                        name: e.data?.tool_name || 'unknown',
                        status: e.data?.success ? 'success' : 'error',
                        duration: e.data?.duration_ms,
                        spanId: s.span_id,
                    });
                }
            }
        });
    });
    return Array.from(toolsMap.values());
}

// ============================================================
// Narrative Summary
// ============================================================
function computeSummary() {
    const traces = Object.values(state.traces);
    if (traces.length === 0) {
        return { total: 0, passed: 0, failed: 0, faults: 0, calls: 0, 
                 resilienceRate: null, avgLatency: null, totalDuration: null, chaosScenarios: 0 };
    }

    let passed = 0, failed = 0, faults = 0, calls = 0, chaosScenarios = 0;
    let latencies = [], totalDuration = 0;

    traces.forEach(t => {
        const report = t.report || {};
        if (t.status === 'success' || report.passed) passed++;
        else failed++;
        
        if ((t.fault_count || 0) > 0) chaosScenarios++;
        faults += t.fault_count || 0;
        calls += t.total_calls || 0;
        
        const elapsed = report.elapsed_s || report.scorecard?.elapsed_s;
        if (elapsed) totalDuration += elapsed;
        
        (t.spans || []).forEach(s => {
            if (s.latency_ms != null) latencies.push(s.latency_ms);
        });
    });

    const avgLatency = latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : null;
    const chaosTraces = traces.filter(t => (t.fault_count || 0) > 0);
    const resilientTraces = chaosTraces.filter(t => t.status === 'success' || t.report?.passed);
    const resilienceRate = chaosTraces.length > 0 ? Math.round((resilientTraces.length / chaosTraces.length) * 100) : null;

    return { total: traces.length, passed, failed, faults, calls, 
             resilienceRate, avgLatency, totalDuration, chaosScenarios };
}

function renderNarrativeSummary() {
    const s = computeSummary();
    const el = document.getElementById('summaryNarrative');
    
    if (s.total === 0) {
        el.innerHTML = 'Waiting for chaos runs...';
        document.getElementById('scenarioCount').textContent = '0';
        return;
    }

    // Build a narrative story
    let narrative = '';
    
    // Verdict
    if (s.failed === 0) {
        narrative += `<span class="pass">✓ All ${s.total} scenarios passed.</span> `;
    } else if (s.passed === 0) {
        narrative += `<span class="fail">✗ All ${s.total} scenarios failed.</span> `;
    } else {
        narrative += `<span class="num">${s.passed}/${s.total}</span> scenarios passed. `;
    }
    
    // Chaos story
    if (s.chaosScenarios > 0) {
        narrative += `Injected <span class="chaos">${s.faults} faults</span> across ${s.chaosScenarios} chaos scenarios. `;
        if (s.resilienceRate !== null) {
            if (s.resilienceRate === 100) {
                narrative += `Agent <span class="pass">survived all chaos</span>. `;
            } else if (s.resilienceRate >= 80) {
                narrative += `<span class="num">${s.resilienceRate}%</span> resilience rate. `;
            } else {
                narrative += `Only <span class="fail">${s.resilienceRate}%</span> resilience. `;
            }
        }
    } else {
        narrative += `<span class="dim">No chaos injected (baseline runs only).</span> `;
    }
    
    // Performance
    if (s.calls > 0) {
        const avgStr = s.avgLatency ? formatLatency(s.avgLatency) : '—';
        const durationStr = s.totalDuration > 0 ? formatDuration(s.totalDuration) : '—';
        narrative += `<span class="dim">${s.calls} LLM calls, ${avgStr} avg, ${durationStr} total.</span>`;
    }
    
    el.innerHTML = narrative;
    document.getElementById('scenarioCount').textContent = s.total;
}

// ============================================================
// Scenario Card
// ============================================================
function getChaosTypeBadge(trace) {
    const report = trace.report || {};
    const meta = report.meta || {};
    const kind = meta.kind || 'none';
    const chaosType = meta.chaos_type || '';
    
    let label = 'BASELINE';
    let cssClass = 'none';
    
    if (kind === 'llm') {
        label = chaosType ? `LLM·${chaosType.replace(/_/g, ' ')}` : 'LLM';
        cssClass = 'llm';
    } else if (kind === 'tool') {
        label = chaosType ? `TOOL·${chaosType}` : 'TOOL';
        cssClass = 'tool';
    } else if (kind === 'context') {
        label = chaosType ? `CTX·${chaosType}` : 'CTX';
        cssClass = 'context';
    }

    return `<span class="chaos-type-badge ${cssClass}">${escapeHtml(label)}</span>`;
}

function renderScenarioCard(trace) {
    const report = trace.report || {};
    const passed = trace.status === 'success' || report.passed;
    const isRunning = trace.status === 'running';
    const hasChaos = (trace.fault_count || 0) > 0;

    let cardClass = 'scenario-card';
    if (isRunning) cardClass += ' running';
    else if (!hasChaos) cardClass += ' baseline';
    else if (passed) cardClass += ' resilient';
    else cardClass += ' failed';

    const elapsedS = report.elapsed_s || report.scorecard?.elapsed_s;
    const tools = extractTools(trace);

    return `
        <div class="${cardClass}" data-trace-id="${trace.trace_id}">
            <div class="card-header">
                <div class="card-identity">
                    <div class="card-name">
                        ${escapeHtml(trace.name)}
                        <code>${trace.trace_id.substring(0, 8)}</code>
                    </div>
                    <div class="card-meta">${getChaosTypeBadge(trace)}</div>
                </div>
                <div class="card-outcome">
                    <span class="outcome-badge ${isRunning ? 'running' : (passed ? 'pass' : 'fail')}">
                        ${isRunning ? 'RUN' : (passed ? 'PASS' : 'FAIL')}
                    </span>
                    <span class="outcome-time">${elapsedS ? formatDuration(elapsedS) : '—'}</span>
                </div>
            </div>
            <div class="card-body">
                <div class="card-stats">
                    <span class="${(trace.fault_count || 0) > 0 ? 'fault' : ''}">⚡${trace.fault_count || 0}</span>
                    <span>📡${trace.total_calls || 0}</span>
                    <span class="${(trace.failed_calls || 0) > 0 ? 'error' : ''}">❌${trace.failed_calls || 0}</span>
                </div>
                ${tools.length > 0 ? `
                    <div class="card-tools">
                        ${tools.slice(0, 4).map(t => `<span class="tool-pill">🛠${escapeHtml(t.name)}</span>`).join('')}
                        ${tools.length > 4 ? `<span class="tool-pill">+${tools.length - 4}</span>` : ''}
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// ============================================================
// Filtering & Rendering
// ============================================================
function applyFilter(filter) {
    state.filter = filter;
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.filter === filter);
    });
    renderScenarios();
}

function getFilteredTraces() {
    const traces = Object.values(state.traces);
    switch (state.filter) {
        case 'passed': return traces.filter(t => t.status === 'success' || t.report?.passed);
        case 'failed': return traces.filter(t => t.status === 'error' || (t.report && !t.report.passed));
        case 'chaos': return traces.filter(t => (t.fault_count || 0) > 0);
        default: return traces;
    }
}

function renderScenarios() {
    const grid = document.getElementById('scenariosGrid');
    const traces = getFilteredTraces();
    
    if (traces.length === 0) {
        const isEmpty = Object.keys(state.traces).length === 0;
        grid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">${isEmpty ? '🎲' : '🔍'}</div>
                <h3 class="empty-title">${isEmpty ? 'Awaiting Chaos...' : 'No matching scenarios'}</h3>
                <p class="empty-text">${isEmpty ? 'Run your agent with chaos scenarios or use <code>chaos_context("test", emit_events=True)</code>' : 'Try a different filter.'}</p>
            </div>
        `;
        return;
    }
    
    traces.sort((a, b) => {
        const timeA = a.end_time || a.start_time || '';
        const timeB = b.end_time || b.start_time || '';
        return timeB.localeCompare(timeA);
    });
    
    grid.innerHTML = traces.map(t => renderScenarioCard(t)).join('');
    
    grid.querySelectorAll('.scenario-card').forEach(card => {
        card.addEventListener('click', () => openScenarioModal(card.dataset.traceId));
    });
}

function render() {
    renderNarrativeSummary();
    renderScenarios();
}

function renderIfChanged() {
    const newHash = computeTracesHash();
    if (newHash !== state.tracesHash) {
        state.tracesHash = newHash;
        render();
    }
}

// ============================================================
// Modal
// ============================================================
function openScenarioModal(traceId) {
    const trace = state.traces[traceId];
    if (!trace) return;
    
    state.selectedTraceId = traceId;
    const modal = document.getElementById('scenarioModal');
    const report = trace.report || {};
    const passed = trace.status === 'success' || report.passed;
    
    document.getElementById('modalTitle').innerHTML = `
        ${escapeHtml(trace.name)}
        <span class="outcome-badge ${passed ? 'pass' : 'fail'}">${passed ? 'PASS' : 'FAIL'}</span>
    `;
    document.getElementById('modalSubtitle').textContent = trace.trace_id;
    
    document.getElementById('modalBody').innerHTML = renderModalContent(trace);
    
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeScenarioModal() {
    document.getElementById('scenarioModal').classList.add('hidden');
    document.body.style.overflow = '';
    state.selectedTraceId = null;
}

function renderModalContent(trace) {
    const report = trace.report || {};
    const faults = extractFaults(trace);
    const tools = extractTools(trace);
    const assertions = report.assertion_results || [];
    const spans = trace.spans || [];
    
    let html = '';
    
    // Agent I/O - Always visible
    html += `
        <div class="detail-section">
            <div class="detail-section-title">Agent I/O</div>
            <div class="agent-io">
                <div class="io-box">
                    <div class="io-label">Input</div>
                    <div class="io-content ${report.agent_input ? '' : 'empty'}">
                        ${report.agent_input ? escapeHtml(report.agent_input) : 'No input captured'}
                    </div>
                </div>
                <div class="io-box">
                    <div class="io-label">Output</div>
                    <div class="io-content ${report.agent_output ? '' : 'empty'}">
                        ${report.agent_output ? escapeHtml(report.agent_output) : 'No output captured'}
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Injected Chaos - Horizontal list like assertions
    if (faults.length > 0) {
        html += `
            <div class="detail-section">
                <div class="detail-section-title">Injected Chaos</div>
                <div class="chaos-list">
                    ${faults.map(f => {
                        const shortSpan = f.spanId.split('_').slice(-1)[0] || 'call';
                        return `
                            <div class="chaos-row">
                                <span class="chaos-icon">⚡</span>
                                <span class="chaos-desc">${escapeHtml(f.type)}</span>
                                <span class="chaos-target">${shortSpan}</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }
    
    // Tools Invoked - Horizontal list like assertions
    if (tools.length > 0) {
        html += `
            <div class="detail-section">
                <div class="detail-section-title">Tools Invoked</div>
                <div class="tools-list">
                    ${tools.map(t => `
                        <div class="tool-row ${t.status}">
                            <span class="tool-icon">🛠</span>
                            <span class="tool-name">${escapeHtml(t.name)}</span>
                            <span class="tool-status ${t.status}">${t.status === 'success' ? '✓' : '✗'}</span>
                            ${t.duration ? `<span class="tool-duration">${formatLatency(t.duration)}</span>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // Assertions
    if (assertions.length > 0) {
        html += `
            <div class="detail-section">
                <div class="detail-section-title">Assertions</div>
                <div class="assertions-list">
                    ${assertions.map(a => `
                        <div class="assertion-row">
                            <div class="assertion-status ${a.passed ? 'pass' : 'fail'}">${a.passed ? '✓' : '✗'}</div>
                            <span class="assertion-name">${escapeHtml(a.name)}</span>
                            ${a.message ? `<span class="assertion-message">${escapeHtml(a.message)}</span>` : ''}
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    // Flame Graph - Separate rows for LLM and Tools
    if (spans.length > 0) {
        html += `
            <div class="detail-section">
                <div class="detail-section-title">Execution Timeline</div>
                ${renderFlameGraph(trace)}
            </div>
        `;
    }
    
    return html;
}

// ============================================================
// Flame Graph - Separate rows for LLM calls and tools
// ============================================================
function renderFlameGraph(trace) {
    const spans = trace.spans || [];
    if (spans.length === 0) return '<div class="io-content empty">No spans recorded.</div>';
    
    // Calculate total duration including tool execution
    let totalDuration = 0;
    const spanData = [];
    
    spans.forEach((span, idx) => {
        const duration = span.latency_ms || 0;
        const faultEvents = (span.events || []).filter(e => e.type === 'fault_injected');
        const toolEvents = (span.events || []).filter(e => e.type === 'tool_end');
        
        spanData.push({
            type: 'llm',
            label: `LLM call ${idx + 1}`,
            duration,
            status: span.status,
            faults: faultEvents,
            offset: totalDuration,
        });
        
        totalDuration += duration;
        
        // Add tool rows nested under this span
        toolEvents.forEach(t => {
            const toolDuration = t.data?.duration_ms || 50; // Default if missing
            spanData.push({
                type: 'tool',
                label: t.data?.tool_name || 'tool',
                duration: toolDuration,
                status: t.data?.success ? 'success' : 'error',
                offset: totalDuration - duration + (duration * 0.1), // Slightly offset
                nested: true,
            });
        });
    });
    
    if (totalDuration === 0) return '<div class="io-content empty">No timing data.</div>';
    
    return `
        <div class="flame-container">
            <div class="flame-graph">
                <div class="flame-legend">
                    <div class="flame-legend-item"><div class="flame-legend-dot llm"></div> LLM Call</div>
                    <div class="flame-legend-item"><div class="flame-legend-dot tool"></div> Tool</div>
                    <div class="flame-legend-item"><div class="flame-legend-dot fault"></div> Fault</div>
                </div>
                <div class="flame-time-axis">
                    <span>0ms</span>
                    <span>${formatLatency(totalDuration / 2)}</span>
                    <span>${formatLatency(totalDuration)}</span>
                </div>
                <div class="flame-rows">
                    ${spanData.map(item => {
                        const leftPercent = (item.offset / totalDuration) * 100;
                        const widthPercent = Math.max((item.duration / totalDuration) * 100, 2);
                        
                        let barClass = item.type;
                        if (item.status === 'error') barClass = 'error';
                        
                        const rowClass = item.nested ? 'flame-row nested' : 'flame-row';
                        const labelPrefix = item.nested ? '└ ' : '';
                        
                        // Fault markers
                        let faultMarkers = '';
                        if (item.faults && item.faults.length > 0) {
                            item.faults.forEach((f, i) => {
                                faultMarkers += `<span class="flame-fault-marker" style="left: ${10 + i * 15}%" title="${escapeHtml(f.data?.fault_type || 'fault')}">⚡</span>`;
                            });
                        }
                        
                        return `
                            <div class="${rowClass}">
                                <div class="flame-row-label">${labelPrefix}${escapeHtml(item.label)}</div>
                                <div class="flame-row-track">
                                    <div class="flame-bar ${barClass}" 
                                         style="left: ${leftPercent}%; width: ${widthPercent}%;"
                                         title="${escapeHtml(item.label)}: ${formatLatency(item.duration)}">
                                        ${widthPercent > 12 ? formatLatency(item.duration) : ''}
                                        ${faultMarkers}
                                    </div>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        </div>
    `;
}

// ============================================================
// Event Handling
// ============================================================
function handleEvent(event) {
    switch (event.type) {
        case 'trace_start':
            state.traces[event.trace_id] = {
                trace_id: event.trace_id,
                name: event.trace_name,
                start_time: event.timestamp,
                status: 'running',
                total_calls: 0, failed_calls: 0, fault_count: 0,
                spans: [],
            };
            break;
            
        case 'trace_end':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                trace.end_time = event.timestamp;
                trace.status = trace.spans.some(s => s.status === 'error') ? 'error' : 'success';
                if (event.data) {
                    trace.total_calls = event.data.total_calls;
                    trace.fault_count = event.data.fault_count;
                }
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
                    if (!event.data?.success) trace.failed_calls++;
                }
            }
            break;
            
        case 'fault_injected':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                trace.fault_count = (trace.fault_count || 0) + 1;
                const span = trace.spans.find(s => s.span_id === event.span_id);
                if (span) span.events.push(event);
            }
            break;
            
        case 'tool_end':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                const span = trace.spans.find(s => s.span_id === event.span_id);
                if (span) span.events.push(event);
            }
            break;
    }
    
    renderIfChanged();
}

// ============================================================
// WebSocket
// ============================================================
function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    ws.onopen = () => {
        document.getElementById('statusBadge').className = 'status-indicator connected';
        document.getElementById('statusText').textContent = 'Live';
    };
    
    ws.onclose = () => {
        document.getElementById('statusBadge').className = 'status-indicator disconnected';
        document.getElementById('statusText').textContent = 'Offline';
        setTimeout(connect, 2000);
    };
    
    ws.onerror = () => {
        document.getElementById('statusBadge').className = 'status-indicator disconnected';
        document.getElementById('statusText').textContent = 'Error';
    };
    
    ws.onmessage = (msg) => {
        try {
            handleEvent(JSON.parse(msg.data));
        } catch (e) {
            console.error('Parse error:', e);
        }
    };
}

// ============================================================
// Init
// ============================================================
function init() {
    initTheme();
    
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
    
    document.querySelectorAll('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => applyFilter(tab.dataset.filter));
    });
    
    document.getElementById('modalClose').addEventListener('click', closeScenarioModal);
    document.getElementById('modalBackdrop').addEventListener('click', closeScenarioModal);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeScenarioModal();
    });
    
    // Load existing traces
    fetch('/api/traces?include_artifacts=true')
        .then(r => r.json())
        .then(traces => {
            traces.forEach(trace => { state.traces[trace.trace_id] = trace; });
            state.tracesHash = computeTracesHash();
            render();
        })
        .catch(err => console.error('Load error:', err));
    
    // Poll with hash-based change detection
    setInterval(() => {
        fetch('/api/traces?include_artifacts=true')
            .then(r => r.json())
            .then(traces => {
                traces.forEach(trace => { state.traces[trace.trace_id] = trace; });
                renderIfChanged();
            })
            .catch(() => {});
    }, 3000);
    
    connect();
    render();
}

document.addEventListener('DOMContentLoaded', init);
