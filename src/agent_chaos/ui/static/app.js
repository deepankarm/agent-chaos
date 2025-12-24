/**
 * agent-chaos dashboard application
 * Version: 2.0 - Resilience metrics + improved event stream
 */

console.log('🃏 agent-chaos dashboard v2.0 - JavaScript file loaded');

const state = {
    traces: {},
    maxLatency: 1000,
    theme: localStorage.getItem('theme') || 'light',
    ui: {
        expandedTraces: new Set(),
        expandedSpans: {}, // trace_id -> Set(span_id)
    },
};

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

function formatTime(isoString) {
    const date = new Date(isoString);
    return date.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit' 
    });
}

function formatLatency(ms) {
    if (ms === null || ms === undefined) return '—';
    if (ms < 1000) return Math.round(ms) + 'ms';
    return (ms / 1000).toFixed(2) + 's';
}

function getEventIcon(type) {
    switch (type) {
        case 'fault_injected': return '⚡';
        case 'ttft': return '⏱';
        case 'stream_cut': return '✂️';
        case 'token_usage': return '🔢';
        case 'tool_use': return '🛠';
        case 'tool_start': return '▶︎';
        case 'tool_end': return '■';
        case 'stream_stats': return '〰️';
        default: return '•';
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function getTraceExpanded(traceId, traceStatus) {
    if (state.ui.expandedTraces.has(traceId)) return true;
    // auto-expand running traces
    return traceStatus === 'running';
}

function toggleTraceExpanded(traceId) {
    if (state.ui.expandedTraces.has(traceId)) state.ui.expandedTraces.delete(traceId);
    else state.ui.expandedTraces.add(traceId);
}

function isSpanExpanded(traceId, spanId) {
    const set = state.ui.expandedSpans[traceId];
    return !!set && set.has(spanId);
}

function toggleSpanExpanded(traceId, spanId) {
    if (!state.ui.expandedSpans[traceId]) state.ui.expandedSpans[traceId] = new Set();
    const set = state.ui.expandedSpans[traceId];
    if (set.has(spanId)) set.delete(spanId);
    else set.add(spanId);
}

function summarizeFaultsFromSpans(spans) {
    const counts = {};
    spans.forEach(s => {
        (s.events || []).forEach(e => {
            if (e.type === 'fault_injected') {
                const ft = e.data?.fault_type || 'unknown';
                counts[ft] = (counts[ft] || 0) + 1;
            }
        });
    });
    return counts;
}

function renderKeyValueTable(rows) {
    return `
        <div class="kv">
            ${rows.map(r => `
                <div class="kv-row">
                    <div class="kv-k">${escapeHtml(r.k)}</div>
                    <div class="kv-v">${r.v}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderContract(report) {
    if (!report) {
        return `<div class="muted">No contract data yet (live trace). For artifact runs, assertions are shown here.</div>`;
    }
    const results = report.assertion_results || [];
    if (results.length === 0) {
        return `<div class="muted">No assertions recorded.</div>`;
    }
    return `
        <div class="table">
            <div class="table-header">
                <div>Assertion</div>
                <div>Status</div>
                <div>Message</div>
            </div>
            ${results.map(r => `
                <div class="table-row">
                    <div><code>${escapeHtml(r.name)}</code></div>
                    <div class="${r.passed ? 'ok' : 'bad'}">${r.passed ? 'PASS' : 'FAIL'}</div>
                    <div class="muted">${escapeHtml(r.message || '')}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function renderChaos(spans, report) {
    const counts = summarizeFaultsFromSpans(spans);
    const types = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    const top = types.slice(0, 6);
    const faultList = top.length
        ? `<div class="pill-row">${top.map(t => `<span class="pill fault">${escapeHtml(t)} × ${counts[t]}</span>`).join('')}</div>`
        : `<div class="muted">No injected faults observed in events.</div>`;
    const expected = report?.scorecard?.faults_injected_total;
    return `
        ${renderKeyValueTable([
            { k: 'Faults observed', v: `${Object.values(counts).reduce((a, b) => a + b, 0)}` },
            { k: 'Faults expected (scorecard)', v: `${expected !== undefined ? expected : '—'}` },
        ])}
        ${faultList}
    `;
}

function sparkline(values, color, markers, opts) {
    const w = opts?.w ?? 160;
    const h = opts?.h ?? 34;
    const vs = values; // preserve index alignment (nulls allowed)
    const vsValid = vs.filter(v => v !== null && v !== undefined);
    if (vsValid.length < 2) return `<div class="muted">—</div>`;
    const min = Math.min(...vsValid);
    const max = Math.max(...vsValid);
    const span = Math.max(1e-9, max - min);
    
    // Margins for axes
    const marginLeft = 45;
    const marginBottom = 25;
    const marginTop = 5;
    const marginRight = 5;
    const plotW = w - marginLeft - marginRight;
    const plotH = h - marginTop - marginBottom;
    
    const pts = vs.map((v, i) => {
        if (v === null || v === undefined) return null;
        const x = marginLeft + (i / (vs.length - 1)) * plotW;
        const y = marginTop + plotH - ((v - min) / span) * plotH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).filter(Boolean).join(' ');
    
    const ms = (markers || []).map(m => {
        const i = m.i;
        if (i === null || i === undefined) return '';
        const x = marginLeft + (i / (vs.length - 1)) * plotW;
        const v = vs[Math.min(vs.length - 1, Math.max(0, i))];
        const y = marginTop + plotH - ((v - min) / span) * plotH;
        const title = escapeHtml(m.title || '');
        const c = m.color || 'var(--warning)';
        return `<g><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.8" fill="${c}"><title>${title}</title></circle></g>`;
    }).join('');
    
    // Axes
    const axisColor = '#444';
    const axisStroke = 1;
    const axisPath = `M${marginLeft},${marginTop} L${marginLeft},${h - marginBottom} L${w - marginRight},${h - marginBottom}`;
    
    // Y-axis labels
    const yLabelStyle = 'font-size:9px;fill:#999;font-family:monospace;';
    const yMax = `<text x="${marginLeft - 5}" y="${marginTop + 10}" text-anchor="end" style="${yLabelStyle}">${Math.round(max)}</text>`;
    const yMin = `<text x="${marginLeft - 5}" y="${h - marginBottom + 4}" text-anchor="end" style="${yLabelStyle}">${Math.round(min)}</text>`;
    
    // X-axis labels (first and last indices)
    const xLabelStyle = 'font-size:9px;fill:#999;font-family:monospace;';
    const xFirst = `<text x="${marginLeft}" y="${h - marginBottom + 15}" text-anchor="middle" style="${xLabelStyle}">0</text>`;
    const xLast = `<text x="${w - marginRight}" y="${h - marginBottom + 15}" text-anchor="middle" style="${xLabelStyle}">${vs.length - 1}</text>`;
    
    return `
        <svg class="spark" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">
            <path d="${axisPath}" stroke="${axisColor}" stroke-width="${axisStroke}" fill="none" />
            ${yMax}
            ${yMin}
            ${xFirst}
            ${xLast}
            <polyline fill="none" stroke="${color}" stroke-width="2" points="${pts}" />
            ${ms}
        </svg>
    `;
}

function summarizeSeries(values) {
    const vs = (values || []).filter(v => v !== null && v !== undefined);
    if (!vs.length) return null;
    return {
        min: Math.min(...vs),
        max: Math.max(...vs),
        last: vs[vs.length - 1],
        n: vs.length,
    };
}

function renderMetrics(spans) {
    const lat = spans.map(s => s.latency_ms).filter(v => v !== null && v !== undefined);
    const ttftBySpan = spans.map(s => {
        const ev = (s.events || []).find(e => e.type === 'ttft' && e.data?.ttft_ms !== undefined);
        return ev ? ev.data.ttft_ms : null;
    }).filter(v => v !== null && v !== undefined);
    // Keep alignment with call index (don't filter), otherwise markers drift.
    const inputTokensRaw = spans.map(s => {
        const ev = (s.events || []).find(e => e.type === 'token_usage' && e.data?.input_tokens !== undefined);
        return ev ? ev.data.input_tokens : null;
    });

    // Markers: place a tool marker on the call that INCLUDED the tool_result.
    const callIdToIndex = new Map();
    spans.forEach((s, idx) => callIdToIndex.set(s.span_id, idx));
    const tokenMarkers = [];
    spans.forEach(s => {
        (s.events || []).forEach(e => {
            if (e.type !== 'tool_end') return;
            const resolvedIn = e.data?.resolved_in_call_id;
            const tool = e.data?.tool_name || 'tool';
            if (!resolvedIn) return;
            const idx = callIdToIndex.get(resolvedIn);
            if (idx === undefined) return;
            tokenMarkers.push({ i: idx, title: `tool_result included: ${tool}`, color: 'var(--warning)' });
        });
    });
    const big = { w: 450, h: 120 };
    const latencySummary = summarizeSeries(lat);
    const ttftSummary = summarizeSeries(ttftBySpan);
    const inputTokSummary = summarizeSeries(inputTokensRaw);
    return `
        <div class="metrics-grid">
            <div class="metric-card" data-metric="latency">
                <div class="metric-title">Latency (ms)</div>
                ${sparkline(lat, 'var(--info)', [], big)}
                <div class="metric-sub">
                    ${latencySummary ? `min=${Math.round(latencySummary.min)} max=${Math.round(latencySummary.max)} last=${Math.round(latencySummary.last)}` : '—'}
                </div>
            </div>
            <div class="metric-card" data-metric="ttft">
                <div class="metric-title">TTFT (ms)</div>
                ${sparkline(ttftBySpan, 'var(--warning)', [], big)}
                <div class="metric-sub">
                    ${ttftSummary ? `min=${Math.round(ttftSummary.min)} max=${Math.round(ttftSummary.max)} last=${Math.round(ttftSummary.last)}` : '—'}
                </div>
            </div>
            <div class="metric-card" data-metric="input_tokens">
                <div class="metric-title">Input tokens (per call)</div>
                ${sparkline(inputTokensRaw, 'var(--success)', tokenMarkers, big)}
                <div class="metric-sub">
                    ${inputTokSummary ? `min=${Math.round(inputTokSummary.min)} max=${Math.round(inputTokSummary.max)} last=${Math.round(inputTokSummary.last)}` : '—'}
                </div>
            </div>
        </div>
        <div class="muted">Click a graph to expand. Tool markers indicate the call where a tool_result was first included.</div>
    `;
}

function renderTools(spans, trace) {
    // Build a clean "invocations" view: 1 row per tool_use_id.
    const inv = new Map(); // tool_use_id -> {tool_name, requested_in, resolved_in, args_bytes, result_bytes, duration_ms, success}
    function upsert(id, patch) {
        if (!id) return;
        const cur = inv.get(id) || { tool_use_id: id };
        inv.set(id, { ...cur, ...patch });
    }

    spans.forEach(s => {
        (s.events || []).forEach(e => {
            if (e.type === 'tool_start') {
                upsert(e.data?.tool_use_id || 'unknown', {
                    tool_name: e.data?.tool_name,
                    requested_in: s.span_id,
                    args_bytes: e.data?.input_bytes,
                    llm_args_ms: e.data?.llm_args_ms,
                });
            }
            if (e.type === 'tool_end') {
                upsert(e.data?.tool_use_id || 'unknown', {
                    tool_name: e.data?.tool_name,
                    resolved_in: e.data?.resolved_in_call_id,
                    result_bytes: e.data?.output_bytes,
                    duration_ms: e.data?.duration_ms,
                    success: e.data?.success,
                    error: e.data?.error,
                });
            }
        });
    });

    // Trace-level tool events (rare; can't correlate to a span_id)
    (trace?.tool_events || []).forEach(e => {
        if (e.type === 'tool_start') {
            upsert(e.data?.tool_use_id || 'unknown', {
                tool_name: e.data?.tool_name,
                args_bytes: e.data?.input_bytes,
                llm_args_ms: e.data?.llm_args_ms,
            });
        }
        if (e.type === 'tool_end') {
            upsert(e.data?.tool_use_id || 'unknown', {
                tool_name: e.data?.tool_name,
                resolved_in: e.data?.resolved_in_call_id,
                result_bytes: e.data?.output_bytes,
                duration_ms: e.data?.duration_ms,
                success: e.data?.success,
                error: e.data?.error,
            });
        }
    });

    const rows = Array.from(inv.values());
    if (rows.length === 0) return `<div class="muted">No tool invocations observed.</div>`;
    rows.sort((a, b) => (a.requested_in || '').localeCompare(b.requested_in || ''));

    return `
        <div class="table">
            <div class="table-header">
                <div>Tool</div>
                <div>Tool call id</div>
                <div>Status</div>
                <div>Generation time</div>
                <div>Args (B)</div>
                <div>Execution time</div>
                <div>Result (B)</div>
                <div>Requested in</div>
                <div>Resolved in</div>
            </div>
            ${rows.map(r => {
                const name = r.tool_name || '—';
                const toolId = r.tool_use_id || '—';
                const statusCls = r.success === true ? 'ok' : (r.success === false ? 'bad' : 'muted');
                const statusTxt = r.success === true ? 'OK' : (r.success === false ? 'ERR' : '—');
                const llmArgs = r.llm_args_ms !== undefined && r.llm_args_ms !== null ? `${Math.round(r.llm_args_ms)}ms` : '—';
                const argsB = r.args_bytes !== undefined && r.args_bytes !== null ? `${r.args_bytes}` : '—';
                const execMs = r.duration_ms !== undefined && r.duration_ms !== null ? `${Math.round(r.duration_ms)}ms` : '—';
                const resB = r.result_bytes !== undefined && r.result_bytes !== null ? `${r.result_bytes}` : '—';
                const req = r.requested_in ? r.requested_in.split('_').slice(0, 2).join('_') : '—';
                const resolvedIn = r.resolved_in ? r.resolved_in.split('_').slice(0, 2).join('_') : '—';
                return `
                    <div class="table-row">
                        <div><code>${escapeHtml(name)}</code></div>
                        <div class="muted"><code>${escapeHtml(toolId)}</code></div>
                        <div class="${statusCls}">${escapeHtml(statusTxt)}</div>
                        <div class="muted">${escapeHtml(llmArgs)}</div>
                        <div class="muted">${escapeHtml(argsB)}</div>
                        <div class="muted">${escapeHtml(execMs)}</div>
                        <div class="muted">${escapeHtml(resB)}</div>
                        <div class="muted"><code>${escapeHtml(req)}</code></div>
                        <div class="muted"><code>${escapeHtml(resolvedIn)}</code></div>
                    </div>
                `;
            }).join('')}
        </div>
    `;
}

function renderTrace(trace) {
    const container = document.getElementById('traces');
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();
    
    let el = document.getElementById('trace-' + trace.trace_id);
    if (!el) {
        el = document.createElement('div');
        el.id = 'trace-' + trace.trace_id;
        el.className = 'trace';
        container.insertBefore(el, container.firstChild);
    }
    
    const spans = trace.spans || [];
    const maxLatency = Math.max(state.maxLatency, ...spans.map(s => s.latency_ms || 0));
    state.maxLatency = maxLatency;
    
    // --- Evidence model ---
    // This UI distinguishes:
    // - Agent/Scenario outcome (contract): comes from RunReport (artifact traces)
    // - LLM-span telemetry: success/failure of patched provider calls
    const report = trace.report || null;
    const hasChaos = (trace.fault_count || 0) > 0;
    const totalCalls = (trace.total_calls !== undefined && trace.total_calls !== null)
        ? trace.total_calls
        : spans.length;
    const failedCalls = (trace.failed_calls !== undefined && trace.failed_calls !== null)
        ? trace.failed_calls
        : spans.filter(s => s.status === 'error').length;
    const successfulCalls = Math.max(0, totalCalls - failedCalls);
    const llmSuccessRate = totalCalls > 0 ? Math.round((successfulCalls / totalCalls) * 100) : 0;

    const scenarioPassed = report ? !!report.passed : (trace.status === 'success');
    const agentErrored = report ? !!report.error : false;
    const agentOutcomeText = report
        ? (agentErrored
            ? (scenarioPassed ? 'Errored (expected)' : 'Errored')
            : 'Completed')
        : '—';

    const assertionResults = report?.assertion_results || [];
    const assertionsTotal = assertionResults.length;
    const assertionsFailed = assertionResults.filter(r => !r.passed).length;
    const assertionsSummary = assertionsTotal > 0
        ? `${assertionsTotal - assertionsFailed}/${assertionsTotal}`
        : '—';
    
    // Determine resilience status
    let resilienceStatus = '';
    let resilienceText = '';
    if (trace.status === 'running') {
        resilienceStatus = 'running';
        resilienceText = 'Running...';
    } else if (!report) {
        // Live traces don't yet have agent-level outcome; show LLM telemetry only.
        if (!hasChaos) {
            resilienceStatus = 'no-chaos';
            resilienceText = 'Baseline (no chaos)';
        } else if (failedCalls > 0 && successfulCalls > 0) {
            resilienceStatus = 'partial';
            resilienceText = `LLM errors observed (${failedCalls}/${totalCalls})`;
        } else if (failedCalls === totalCalls && totalCalls > 0) {
            resilienceStatus = 'failed';
            resilienceText = `LLM all failed (${failedCalls}/${totalCalls})`;
        } else {
            resilienceStatus = 'resilient';
            resilienceText = `Chaos injected (LLM ok, ${llmSuccessRate}% success)`;
        }
    } else if (!hasChaos) {
        resilienceStatus = 'no-chaos';
        resilienceText = 'Baseline (no chaos)';
    } else if (!agentErrored && failedCalls > 0) {
        // This is the strongest evidence: agent completed despite injected failures.
        resilienceStatus = 'resilient';
        resilienceText = `Recovered (agent completed with ${failedCalls} LLM error${failedCalls === 1 ? '' : 's'})`;
    } else if (!agentErrored) {
        resilienceStatus = 'resilient';
        resilienceText = 'Completed under chaos';
    } else if (scenarioPassed && agentErrored) {
        // Contract may intentionally assert a failure mode.
        resilienceStatus = 'partial';
        resilienceText = 'Expected failure mode observed';
    } else {
        resilienceStatus = 'failed';
        resilienceText = 'Broke under chaos';
    }

    // Trace dot: if scenario "passes" but agent errored (expected failure), show warning color.
    const statusClass = (report && scenarioPassed && agentErrored) ? 'running' : trace.status;
    const expanded = getTraceExpanded(trace.trace_id, trace.status);
    el.className = `trace ${expanded ? 'expanded' : ''}`;
    
    el.innerHTML = `
        <div class="trace-header" data-trace-id="${trace.trace_id}">
            <div class="trace-title">
                <div class="trace-status ${statusClass}"></div>
                <div class="trace-title-text">
                    <span class="trace-name">${escapeHtml(trace.name)}</span>
                    <span class="trace-id">${trace.trace_id}</span>
                </div>
            </div>
            <div class="trace-stats">
                <div class="trace-stat-group">
                    <div class="trace-stat">
                        <span class="trace-stat-label">Scenario:</span>
                        <span class="trace-stat-value ${scenarioPassed ? 'success' : 'fault'}">${scenarioPassed ? 'PASS' : 'FAIL'}</span>
                    </div>
                    <div class="trace-stat">
                        <span class="trace-stat-label">Agent:</span>
                        <span class="trace-stat-value ${agentErrored ? 'fault' : 'success'}">${escapeHtml(agentOutcomeText)}</span>
                    </div>
                </div>
                <div class="trace-stat-group">
                    <div class="trace-stat">
                        <span class="trace-stat-label">LLM Calls:</span>
                        <span class="trace-stat-value">${totalCalls || 0}</span>
                    </div>
                    <div class="trace-stat">
                        <span class="trace-stat-label">LLM Errors:</span>
                        <span class="trace-stat-value ${failedCalls > 0 ? 'fault' : ''}">${failedCalls}</span>
                    </div>
                </div>
                <div class="trace-stat-group">
                    <div class="trace-stat">
                        <span class="trace-stat-label">Faults Injected:</span>
                        <span class="trace-stat-value fault">${trace.fault_count || 0}</span>
                    </div>
                    <div class="trace-stat">
                        <span class="trace-stat-label">Assertions:</span>
                        <span class="trace-stat-value ${assertionsFailed > 0 ? 'fault' : 'success'}">${assertionsSummary}</span>
                    </div>
                </div>
                <div class="trace-resilience ${resilienceStatus}">
                    ${resilienceText}
                </div>
                <button class="trace-toggle" title="Toggle details" aria-label="Toggle details">
                    <span class="chev">${expanded ? '▾' : '▸'}</span>
                </button>
            </div>
        </div>
        <div class="trace-body ${expanded ? '' : 'collapsed'}">
            <div class="sections">
                <div class="section">
                    <div class="section-header">Contract</div>
                    <div class="section-body">
                        ${renderContract(report)}
                    </div>
                </div>
                <div class="section">
                    <div class="section-header">Chaos</div>
                    <div class="section-body">
                        ${renderChaos(spans, report)}
                    </div>
                </div>
                <div class="section section-full">
                    <div class="section-header">Metrics</div>
                    <div class="section-body">
                        ${renderMetrics(spans)}
                    </div>
                </div>
                <div class="section section-full">
                    <div class="section-header">Tools</div>
                    <div class="section-body">
                        ${renderTools(spans, trace)}
                    </div>
                </div>
            </div>
            <div class="waterfall">
                <div class="waterfall-header">
                    <div>Span ID</div>
                    <div>Provider</div>
                    <div>Timeline</div>
                    <div style="text-align: right">Duration</div>
                </div>
                <div class="timeline-legend">
                    <span class="legend-item"><span style="color: var(--warning)">⚡</span> TTFT</span>
                    <span class="legend-item"><span style="color: var(--info)">▶</span> Tool args gen start</span>
                    <span class="legend-item"><span style="color: var(--success)">✓</span> Tool args gen end</span>
                    <span class="legend-item"><span style="color: var(--accent)">✗</span> Fault</span>
                    <span class="legend-item"><span style="color: var(--accent)">⏸</span> Hang</span>
                    <span class="legend-item"><span style="color: var(--accent)">✂</span> Cut</span>
                </div>
                ${spans.map(span => renderSpan(span, maxLatency, trace.trace_id)).join('')}
            </div>
        </div>
    `;
}

function renderSpan(span, maxLatency, traceId) {
    const barWidth = span.latency_ms 
        ? Math.max(5, (span.latency_ms / maxLatency) * 100) 
        : (span.status === 'running' ? 30 : 5);
    
    const events = span.events || [];
    
    // Build timeline markers positioned by event timing
    const spanDuration = span.latency_ms || 1;
    const spanStartTime = span.start_time;
    const timelineMarkers = [];
    
    // Parse event timestamp to calculate relative position
    const parseEventTime = (e) => {
        if (!e.timestamp) return null;
        try {
            return new Date(e.timestamp).getTime() / 1000; // seconds
        } catch {
            return null;
        }
    };
    
    events.forEach((e, idx) => {
        let marker = null;
        let offset = 50; // Default middle
        
        // Calculate position from timestamps
        const eventTime = parseEventTime(e);
        if (eventTime && spanStartTime) {
            const relativeMs = (eventTime - spanStartTime) * 1000;
            offset = (relativeMs / spanDuration) * 100;
        } else {
            // Fallback heuristics based on event type
            if (e.type === 'ttft' && e.data?.ttft_ms !== undefined) {
                offset = (e.data.ttft_ms / spanDuration) * 100;
            } else if (e.type === 'tool_use') {
                offset = 30;
            } else if (e.type === 'tool_start') {
                offset = 35;
            } else if (e.type === 'fault_injected') {
                offset = 10;
            }
        }
        offset = Math.max(2, Math.min(98, offset)); // Clamp with margins
        
        if (e.type === 'ttft' && e.data?.ttft_ms !== undefined) {
            marker = { offset, color: 'var(--warning)', char: '⚡', title: `TTFT: ${formatLatency(e.data.ttft_ms)}` };
        } else if (e.type === 'tool_use' && e.data?.tool_name) {
            marker = { offset, color: 'var(--info)', char: '▶', title: `Tool args gen start: ${e.data.tool_name}` };
        } else if (e.type === 'tool_start' && e.data?.tool_name) {
            marker = { offset, color: 'var(--success)', char: '✓', title: `Tool args gen end: ${e.data.tool_name}${e.data.llm_args_ms ? ` (${Math.round(e.data.llm_args_ms)}ms)` : ''}` };
        } else if (e.type === 'fault_injected' && e.data?.fault_type) {
            marker = { offset, color: 'var(--accent)', char: '✗', title: `Fault: ${e.data.fault_type}` };
        } else if (e.type === 'stream_hang') {
            marker = { offset, color: 'var(--accent)', char: '⏸', title: 'Stream hang' };
        } else if (e.type === 'stream_cut' && e.data?.chunk_count) {
            marker = { offset, color: 'var(--accent)', char: '✂', title: `Stream cut after ${e.data.chunk_count} chunks` };
        }
        
        if (marker) timelineMarkers.push(marker);
    });
    
    const timelineMarkersHtml = timelineMarkers.map(m => 
        `<div class="timeline-marker" style="left: ${m.offset}%; color: ${m.color};" title="${escapeHtml(m.title)}">${m.char}</div>`
    ).join('');
    
    const shortId = span.span_id.split('_').slice(0, 2).join('_');
    const latencyText = span.error 
        ? '✗ ' + span.error.split(':')[0].substring(0, 20)
        : formatLatency(span.latency_ms);

    const expanded = isSpanExpanded(traceId, span.span_id);
                const faultTypes = Array.from(new Set(events.filter(e => e.type === 'fault_injected').map(e => e.data?.fault_type).filter(Boolean)));
                const toolUses = events.filter(e => e.type === 'tool_use').map(e => e.data?.tool_name).filter(Boolean);
                const usageEv = events.find(e => e.type === 'token_usage');
                const usageText = usageEv
                    ? `${usageEv.data?.input_tokens ?? '—'} in / ${usageEv.data?.output_tokens ?? '—'} out`
                    : '—';
    const ttft = (() => {
        const ev = events.find(e => e.type === 'ttft' && e.data?.ttft_ms !== undefined);
        return ev ? formatLatency(ev.data.ttft_ms) : '—';
    })();
    const rawEvents = events.map(e => {
        const t = e.timestamp ? formatTime(e.timestamp) : '';
        const data = e.data ? JSON.stringify(e.data) : '';
        return `${t} ${e.type}${data ? ' ' + data : ''}`;
    }).join('\n');
    
    return `
        <div class="span-block ${expanded ? 'expanded' : ''}">
            <div class="span-row" data-trace-id="${traceId}" data-span-id="${escapeHtml(span.span_id)}" title="Toggle span details">
                <div class="span-id" title="${escapeHtml(span.span_id)}">${shortId}</div>
                <div class="span-provider">${span.provider}</div>
                <div class="span-bar-container">
                    <div class="span-bar ${span.status}" style="width: ${barWidth}%">
                        ${timelineMarkersHtml}
                    </div>
                </div>
                <div class="span-latency ${span.status}">${latencyText}</div>
            </div>
            <div class="span-details">
                ${renderKeyValueTable([
                    { k: 'Span ID', v: `<code>${escapeHtml(span.span_id)}</code>` },
                    { k: 'Status', v: `<span class="${span.status === 'success' ? 'ok' : (span.status === 'error' ? 'bad' : 'muted')}">${escapeHtml(span.status)}</span>` },
                    { k: 'Latency', v: `<code>${escapeHtml(formatLatency(span.latency_ms))}</code>` },
                    { k: 'TTFT', v: `<code>${escapeHtml(ttft)}</code>` },
                    { k: 'Tokens (in/out)', v: `<code>${escapeHtml(usageText)}</code>` },
                    { k: 'Faults', v: faultTypes.length ? `<div class="pill-row">${faultTypes.map(f => `<span class="pill fault">${escapeHtml(f)}</span>`).join('')}</div>` : `<span class="muted">—</span>` },
                    { k: 'Tool use (requested)', v: toolUses.length ? `<div class="pill-row">${toolUses.map(n => `<span class="pill">${escapeHtml(n)}</span>`).join('')}</div>` : `<span class="muted">—</span>` },
                ])}
                ${span.error ? `<div class="detail-block"><div class="detail-title">Error</div><pre class="pre">${escapeHtml(span.error)}</pre></div>` : ''}
                ${rawEvents ? `<div class="detail-block"><div class="detail-title">Events</div><pre class="pre">${escapeHtml(rawEvents)}</pre></div>` : ''}
            </div>
        </div>
    `;
}

function addLogEntry(event) {
    const log = document.getElementById('logContent');
    const div = document.createElement('div');
    div.className = 'log-entry';
    
    let detail = '';
    if (event.data?.fault_type) detail = event.data.fault_type;
    else if (event.data?.ttft_ms) detail = formatLatency(event.data.ttft_ms);
    else if (event.data?.latency_ms) detail = formatLatency(event.data.latency_ms);
    else if (event.data?.chunk_count) detail = 'after ' + event.data.chunk_count + ' chunks';
    
    const sessionName = event.trace_name || '—';
    const shortTraceId = event.trace_id ? event.trace_id.substring(0, 8) : '—';
    const shortSpanId = event.span_id ? event.span_id.split('_').slice(0, 2).join('_') : '—';
    
    div.innerHTML = `
        <div class="log-time">${formatTime(event.timestamp)}</div>
        <div class="log-session-name" title="${escapeHtml(event.trace_name || '')}">${escapeHtml(sessionName)}</div>
        <div class="log-session-id" title="${escapeHtml(event.trace_id || '')}">${shortTraceId}</div>
        <div class="log-span" title="${escapeHtml(event.span_id || '')}">${shortSpanId}</div>
        <div class="log-type ${event.type}">${event.type.replace(/_/g, ' ')}</div>
        <div class="log-detail">${escapeHtml(detail)}</div>
    `;
    
    // Insert after header (if exists) or at top
    const header = log.querySelector('.log-header');
    if (header) {
        header.insertAdjacentElement('afterend', div);
    } else {
        log.insertBefore(div, log.firstChild);
    }
    
    // Keep only last 50 entries (excluding header)
    const entries = Array.from(log.children).filter(c => !c.classList.contains('log-header'));
    while (entries.length > 50) {
        const last = entries.pop();
        if (last) last.remove();
    }
}

function handleEvent(event) {
    addLogEntry(event);
    
    switch (event.type) {
        case 'trace_start':
            state.traces[event.trace_id] = {
                trace_id: event.trace_id,
                name: event.trace_name,
                start_time: event.timestamp,
                status: 'running',
                total_calls: 0,
                fault_count: 0,
                spans: [],
            };
            renderTrace(state.traces[event.trace_id]);
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
                renderTrace(trace);
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
                renderTrace(trace);
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
                }
                renderTrace(trace);
            }
            break;
            
        case 'fault_injected':
        case 'ttft':
        case 'stream_cut':
        case 'token_usage':
        case 'tool_use':
        case 'tool_start':
        case 'tool_end':
        case 'stream_stats':
            if (state.traces[event.trace_id]) {
                const trace = state.traces[event.trace_id];
                const span = trace.spans.find(s => s.span_id === event.span_id);
                if (span) {
                    span.events.push(event);
                } else {
                    // Tool events may not have a span_id (framework doesn't expose tool_use_id yet).
                    trace.tool_events = trace.tool_events || [];
                    trace.tool_events.push(event);
                }
                if (event.type === 'fault_injected') {
                    trace.fault_count = (trace.fault_count || 0) + 1;
                }
                renderTrace(trace);
            }
            break;
    }
}

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
            handleEvent(event);
        } catch (e) {
            console.error('Failed to parse event:', e);
        }
    };
}

function init() {
    initTheme();

    // Event delegation for expand/collapse
    const tracesEl = document.getElementById('traces');
    tracesEl.addEventListener('click', (e) => {
        const metricCard = e.target.closest('.metric-card');
        if (metricCard) {
            const traceEl = e.target.closest('.trace');
            const traceId = traceEl?.id?.replace('trace-', '') || null;
            if (traceId && state.traces[traceId]) {
                openMetricModal(state.traces[traceId], metricCard.getAttribute('data-metric'));
            }
            return;
        }

        const spanRow = e.target.closest('.span-row');
        if (spanRow) {
            const traceId = spanRow.getAttribute('data-trace-id');
            const spanId = spanRow.getAttribute('data-span-id');
            if (traceId && spanId) {
                toggleSpanExpanded(traceId, spanId);
                const t = state.traces[traceId];
                if (t) renderTrace(t);
            }
            return;
        }

        const header = e.target.closest('.trace-header');
        if (header) {
            const traceId = header.getAttribute('data-trace-id');
            if (traceId) {
                toggleTraceExpanded(traceId);
                const t = state.traces[traceId];
                if (t) renderTrace(t);
            }
        }
    });
    
    fetch('/api/traces')
        .then(r => r.json())
        .then(traces => {
            traces.forEach(trace => {
                state.traces[trace.trace_id] = trace;
                renderTrace(trace);
            });
        })
        .catch(err => console.error('Failed to load traces:', err));
    
    // Poll for artifact traces every few seconds so CLI runs show up even if the
    // dashboard wasn't running during the run.
    setInterval(() => {
        fetch('/api/traces?include_artifacts=true')
            .then(r => r.json())
            .then(traces => {
                traces.forEach(trace => {
                    state.traces[trace.trace_id] = trace;
                    renderTrace(trace);
                });
            })
            .catch(() => {});
    }, 3000);
    
    connect();
}

function ensureModal() {
    let modal = document.getElementById('metricModal');
    if (modal) return modal;
    modal = document.createElement('div');
    modal.id = 'metricModal';
    modal.className = 'modal hidden';
    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-panel">
            <div class="modal-header">
                <div class="modal-title" id="metricModalTitle">Metric</div>
                <button class="modal-close" id="metricModalClose" aria-label="Close">✕</button>
            </div>
            <div class="modal-body" id="metricModalBody"></div>
        </div>
    `;
    document.body.appendChild(modal);
    modal.querySelector('.modal-backdrop').addEventListener('click', closeMetricModal);
    modal.querySelector('#metricModalClose').addEventListener('click', closeMetricModal);
    window.addEventListener('keydown', (ev) => {
        if (ev.key === 'Escape') closeMetricModal();
    });
    return modal;
}

function closeMetricModal() {
    const modal = document.getElementById('metricModal');
    if (modal) modal.classList.add('hidden');
}

function openMetricModal(trace, metricKey) {
    const modal = ensureModal();
    const spans = trace.spans || [];

    const lat = spans.map(s => s.latency_ms);
    const ttft = spans.map(s => {
        const ev = (s.events || []).find(e => e.type === 'ttft' && e.data?.ttft_ms !== undefined);
        return ev ? ev.data.ttft_ms : null;
    });
    const inputTok = spans.map(s => {
        const ev = (s.events || []).find(e => e.type === 'token_usage' && e.data?.input_tokens !== undefined);
        return ev ? ev.data.input_tokens : null;
    });

    const callIdToIndex = new Map();
    spans.forEach((s, idx) => callIdToIndex.set(s.span_id, idx));
    const tokenMarkers = [];
    spans.forEach(s => (s.events || []).forEach(e => {
        if (e.type !== 'tool_end') return;
        const resolvedIn = e.data?.resolved_in_call_id;
        if (!resolvedIn) return;
        const idx = callIdToIndex.get(resolvedIn);
        if (idx === undefined) return;
        tokenMarkers.push({ i: idx, title: `tool_result included: ${e.data?.tool_name || 'tool'}`, color: 'var(--warning)' });
    }));

    let title = 'Metric';
    let series = [];
    let color = 'var(--info)';
    let markers = [];
    if (metricKey === 'latency') { title = 'Latency (ms)'; series = lat; color = 'var(--info)'; }
    else if (metricKey === 'ttft') { title = 'TTFT (ms)'; series = ttft; color = 'var(--warning)'; }
    else { title = 'Input tokens (per call)'; series = inputTok; color = 'var(--success)'; markers = tokenMarkers; }

    const sum = summarizeSeries(series);
    const big = { w: 980, h: 260 };
    document.getElementById('metricModalTitle').textContent = `${trace.name} · ${title}`;
    document.getElementById('metricModalBody').innerHTML = `
        <div class="modal-kv">
            <div><span class="muted">points</span> <b>${sum ? sum.n : 0}</b></div>
            <div><span class="muted">min</span> <b>${sum ? Math.round(sum.min) : '—'}</b></div>
            <div><span class="muted">max</span> <b>${sum ? Math.round(sum.max) : '—'}</b></div>
            <div><span class="muted">last</span> <b>${sum ? Math.round(sum.last) : '—'}</b></div>
        </div>
        <div class="modal-chart">
            ${sparkline(series, color, markers, big)}
        </div>
        <div class="muted">Tip: markers indicate where a tool_result was first included in the next LLM call.</div>
    `;
    modal.classList.remove('hidden');
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('🃏 agent-chaos dashboard v2.0 loaded');
    init();
});

