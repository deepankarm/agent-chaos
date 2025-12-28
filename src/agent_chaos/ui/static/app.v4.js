/**
 * agent-chaos dashboard v4.0
 * Conversation-First Timeline UI
 */

console.log('🃏 agent-chaos dashboard v4.0 loaded');

// ============================================================
// State
// ============================================================
const state = {
    traces: {},
    tracesHash: '',
    theme: localStorage.getItem('theme') || 'dark',
    filter: 'all',
    typeFilter: [], // Array of selected types: ['user_input', 'llm', 'stream', 'tool', 'context']
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

// Mapping for chaos type display names
const CHAOS_TYPE_LABELS = {
    'user_input': 'User Input',
    'llm': 'LLM Call',
    'stream': 'LLM Streaming',
    'tool': 'Tools',
    'context': 'Context',
};

function getChaosTypeLabel(type) {
    return CHAOS_TYPE_LABELS[type] || type.toUpperCase();
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

function formatMs(ms) {
    if (ms == null) return '—';
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

function formatTimestamp(ms) {
    if (ms == null) return '0.0s';
    return `${(ms / 1000).toFixed(1)}s`;
}

function truncateText(text, maxLen = 200) {
    if (!text || text.length <= maxLen) return text;
    return text.substring(0, maxLen) + '...';
}

// Hash to detect real changes
function computeTracesHash() {
    const keys = Object.keys(state.traces).sort();
    const parts = keys.map(k => {
        const t = state.traces[k];
        const convLen = (t.report?.conversation || []).length;
        return `${k}:${t.status}:${t.total_calls}:${t.fault_count}:${convLen}`;
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
                    chaos_point: e.data?.chaos_point || null,  // LLM, STREAM, TOOL, CONTEXT
                    fn_name: e.data?.chaos_fn_name,
                    fn_doc: e.data?.chaos_fn_doc,
                    target_tool: e.data?.target_tool,
                    original: e.data?.original,
                    mutated: e.data?.mutated,
                    // Context mutation details
                    added_messages: e.data?.added_messages,
                    removed_messages: e.data?.removed_messages,
                    added_count: e.data?.added_count,
                    removed_count: e.data?.removed_count,
                    spanId: s.span_id,
                    timestamp: e.timestamp,
                });
            }
        });
    });
    return faults;
}

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
                        result: e.data?.result,
                        spanId: s.span_id,
                    });
                }
            }
        });
    });
    return Array.from(toolsMap.values());
}

// Build conversation from report or events
function buildConversation(trace) {
    const report = trace.report || {};
    
    // If we have a conversation array in the report, use it
    if (report.conversation && report.conversation.length > 0) {
        return report.conversation;
    }
    
    // Otherwise build from events and agent_input/output
    const conversation = [];
    
    // Add user message
    if (report.agent_input) {
        conversation.push({
            type: 'user',
            content: report.agent_input,
            timestamp_ms: 0,
        });
    }
    
    // Extract from spans/events
    (trace.spans || []).forEach(span => {
        (span.events || []).forEach(e => {
            if (e.type === 'tool_use') {
                conversation.push({
                    type: 'tool_call',
                    tool_name: e.data?.tool_name,
                    tool_use_id: e.data?.tool_use_id,
                    args: e.data?.args,
                    timestamp_ms: null, // We don't have precise timing from events
                });
            } else if (e.type === 'tool_end') {
                conversation.push({
                    type: 'tool_result',
                    tool_name: e.data?.tool_name,
                    tool_use_id: e.data?.tool_use_id,
                    result: e.data?.result,
                    success: e.data?.success,
                    duration_ms: e.data?.duration_ms,
                    timestamp_ms: null,
                });
            } else if (e.type === 'fault_injected') {
                conversation.push({
                    type: 'chaos',
                    fault_type: e.data?.fault_type,
                    chaos_fn_name: e.data?.chaos_fn_name,
                    chaos_fn_doc: e.data?.chaos_fn_doc,
                    target_tool: e.data?.target_tool,
                    original: e.data?.original,
                    mutated: e.data?.mutated,
                    // Context mutation details
                    added_messages: e.data?.added_messages,
                    removed_messages: e.data?.removed_messages,
                    added_count: e.data?.added_count,
                    removed_count: e.data?.removed_count,
                    timestamp_ms: null,
                });
            }
        });
    });
    
    // Add assistant message
    if (report.agent_output) {
        conversation.push({
            type: 'assistant',
            content: report.agent_output,
            timestamp_ms: (report.elapsed_s || 0) * 1000,
        });
    }
    
    return conversation;
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
    const summaryBar = document.getElementById('summaryBar');
    const headerStats = document.getElementById('headerStats');
    
    if (s.total === 0) {
        summaryBar.classList.add('hidden');
        headerStats.innerHTML = '';
        document.getElementById('scenarioCount').textContent = '0';
        return;
    }

    // Hide the old summary bar
    summaryBar.classList.add('hidden');
    
    // Build compact header stats
    const passClass = s.failed === 0 ? 'all-pass' : (s.passed === 0 ? 'all-fail' : '');
    const resilienceClass = s.resilienceRate >= 80 ? 'good' : (s.resilienceRate >= 50 ? 'warn' : 'bad');
    
    let statsHtml = `
        <div class="header-stat ${passClass}">
            <span class="stat-value">${s.passed}/${s.total}</span>
            <span class="stat-label">passed</span>
        </div>
        <div class="header-stat chaos">
            <span class="stat-value">⚡${s.faults}</span>
            <span class="stat-label">chaos</span>
        </div>
    `;
    
    if (s.resilienceRate !== null) {
        statsHtml += `
            <div class="header-stat ${resilienceClass}">
                <span class="stat-value">${s.resilienceRate}%</span>
                <span class="stat-label">resilient</span>
            </div>
        `;
    }
    
    if (s.totalDuration > 0) {
        statsHtml += `
            <div class="header-stat">
                <span class="stat-value">${formatDuration(s.totalDuration)}</span>
                <span class="stat-label">total</span>
            </div>
        `;
    }
    
    headerStats.innerHTML = statsHtml;
    document.getElementById('scenarioCount').textContent = s.total;
}

// ============================================================
// Scenario Card
// ============================================================
function getChaosTypeBadge(trace) {
    // Get faults and determine chaos points from backend data
    const faults = extractFaults(trace);
    
    // No chaos
    if (faults.length === 0) {
        return `<span class="chaos-type-badge none">NONE</span>`;
    }
    
    // Get unique chaos points from faults (use backend data)
    const uniquePoints = new Set(faults.map(f => f.chaos_point || getChaosPointFallback(f.type)));
    
    // Multiple types
    if (uniquePoints.size > 1) {
        return `<span class="chaos-type-badge multiple">MULTIPLE</span>`;
    }
    
    // Single type - use the chaos_point and display label
    const point = [...uniquePoints][0] || 'unknown';
    const cssClass = point.toLowerCase();
    const displayLabel = getChaosTypeLabel(cssClass);
    
    return `<span class="chaos-type-badge ${cssClass}">${displayLabel}</span>`;
}

// Fallback for older events without chaos_point field
function getChaosPointFallback(faultType) {
    if (!faultType) return 'UNKNOWN';
    const type = faultType.toLowerCase();
    if (type.includes('user_input') || type.includes('user_mutate')) return 'USER_INPUT';
    if (type.includes('stream') || type.includes('ttft') || type.includes('chunk') || type.includes('hang')) return 'STREAM';
    if (type.includes('tool') || type.includes('mutate')) return 'TOOL';
    if (type.includes('context') || type.includes('truncate') || type.includes('distractor')) return 'CONTEXT';
    // Default to LLM for API errors, timeouts, etc.
    return 'LLM';
}

function renderScenarioCard(trace) {
    const report = trace.report || {};
    const passed = trace.status === 'success' || report.passed;
    const isRunning = trace.status === 'running';

    let cardClass = 'scenario-card';
    if (isRunning) cardClass += ' running';
    else if (passed) cardClass += ' passed';
    else cardClass += ' failed';

    const elapsedS = report.elapsed_s || report.scorecard?.elapsed_s;
    const assertions = report.assertion_results || [];
    const failedAssertions = assertions.filter(a => !a.passed);
    const passedAssertions = assertions.filter(a => a.passed);
    const passedCount = passedAssertions.length;
    const chaosCount = trace.fault_count || 0;
    const description = trace.description || '';

    // Build tooltip content
    const faults = extractFaults(trace);
    const chaosItems = faults.length > 0 
        ? faults.map(f => f.type + (f.target_tool ? ` (${f.target_tool})` : ''))
        : ['No chaos injected'];
    const passedItems = passedAssertions.length > 0
        ? passedAssertions.map(a => a.name)
        : ['None'];
    const failedItems = failedAssertions.length > 0
        ? failedAssertions.map(a => a.name)
        : [];

    // Build inline stats with CSS tooltips
    let statsHtml = '';
    if (!isRunning && assertions.length > 0) {
        const chaosTooltipHtml = `<div class="tooltip"><div class="tooltip-title">Chaos Injected</div>${chaosItems.map(i => `<div class="tooltip-item">${escapeHtml(i)}</div>`).join('')}</div>`;
        const passedTooltipHtml = `<div class="tooltip"><div class="tooltip-title">Passed</div>${passedItems.map(i => `<div class="tooltip-item tooltip-pass">✓ ${escapeHtml(i)}</div>`).join('')}</div>`;
        
        if (failedAssertions.length > 0) {
            const failedTooltipHtml = `<div class="tooltip"><div class="tooltip-title">Failed</div>${failedItems.map(i => `<div class="tooltip-item tooltip-fail">✗ ${escapeHtml(i)}</div>`).join('')}</div>`;
            statsHtml = `<span class="inline-stats"><span class="stat-chaos has-tooltip">⚡${chaosCount}${chaosTooltipHtml}</span><span class="stat-fail has-tooltip">✗${failedAssertions.length}${failedTooltipHtml}</span><span class="stat-pass has-tooltip">✓${passedCount}${passedTooltipHtml}</span></span>`;
        } else {
            statsHtml = `<span class="inline-stats"><span class="stat-chaos has-tooltip">⚡${chaosCount}${chaosTooltipHtml}</span><span class="stat-pass has-tooltip">✓${assertions.length}${passedTooltipHtml}</span></span>`;
        }
    }

    // Build description tooltip for card hover (positioned below to avoid header)
    const descriptionTooltip = description 
        ? `<div class="tooltip tooltip-below"><div class="tooltip-title">Description</div><div class="tooltip-item tooltip-description">${escapeHtml(description)}</div></div>`
        : '';

    return `
        <div class="${cardClass} ${description ? 'has-description' : ''}" data-trace-id="${trace.trace_id}">
            <div class="card-header">
                <div class="card-identity">
                    <div class="card-name ${description ? 'has-tooltip' : ''}">${escapeHtml(trace.name)}${descriptionTooltip}</div>
                    <div class="card-meta">${getChaosTypeBadge(trace)}${statsHtml}</div>
                </div>
                <div class="card-outcome">
                    <span class="outcome-badge ${isRunning ? 'running' : (passed ? 'pass' : 'fail')}">
                        ${isRunning ? 'RUN' : (passed ? 'PASS' : 'FAIL')}
                    </span>
                    <span class="outcome-time">${elapsedS ? formatDuration(elapsedS) : '—'}</span>
                </div>
            </div>
        </div>
    `;
}

// ============================================================
// Filtering & Rendering
// ============================================================
function applyFilter(filter) {
    state.filter = filter;
    document.querySelectorAll('#filterTabs .filter-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.filter === filter);
    });
    renderScenarios();
}

function toggleTypeFilter(type) {
    // Toggle - add if not present, remove if present
    const index = state.typeFilter.indexOf(type);
    if (index > -1) {
        state.typeFilter.splice(index, 1);
    } else {
        state.typeFilter.push(type);
    }
    updateChaosTypeFilterUI();
    renderScenarios();
}

function updateChaosTypeFilterUI() {
    // Update checkboxes
    document.querySelectorAll('#chaosTypeDropdown input[type="checkbox"]').forEach(checkbox => {
        checkbox.checked = state.typeFilter.includes(checkbox.dataset.type);
    });
    
    // Update chips
    const chipsContainer = document.getElementById('chaosTypeChips');
    chipsContainer.innerHTML = '';
    
    state.typeFilter.forEach(type => {
        const chip = document.createElement('span');
        chip.className = 'chaos-type-chip';
        chip.setAttribute('data-type', type);
        chip.innerHTML = `${getChaosTypeLabel(type)} <span class="chip-remove" data-type="${type}">×</span>`;
        chipsContainer.appendChild(chip);
    });
    
    // Update dropdown trigger appearance
    const trigger = document.getElementById('chaosTypeDropdownTrigger');
    if (state.typeFilter.length > 0) {
        trigger.classList.add('has-selection');
    } else {
        trigger.classList.remove('has-selection');
    }
}

function toggleDropdown() {
    const dropdown = document.getElementById('chaosTypeDropdown');
    const trigger = document.getElementById('chaosTypeDropdownTrigger');
    const isOpen = dropdown.classList.toggle('open');
    if (isOpen) {
        trigger.classList.add('dropdown-open');
    } else {
        trigger.classList.remove('dropdown-open');
    }
}

function closeDropdown() {
    const dropdown = document.getElementById('chaosTypeDropdown');
    const trigger = document.getElementById('chaosTypeDropdownTrigger');
    dropdown.classList.remove('open');
    trigger.classList.remove('dropdown-open');
}

function getFilteredTraces() {
    let traces = Object.values(state.traces);
    
    // Apply pass/fail filter
    switch (state.filter) {
        case 'passed': 
            traces = traces.filter(t => t.status === 'success' || t.report?.passed);
            break;
        case 'failed': 
            traces = traces.filter(t => t.status === 'error' || (t.report && !t.report.passed));
            break;
    }
    
    // Apply type filter based on chaos_point from faults (multi-select)
    if (state.typeFilter.length > 0) {
        traces = traces.filter(t => {
            const faults = extractFaults(t);
            const points = faults.map(f => (f.chaos_point || getChaosPointFallback(f.type)).toUpperCase());
            // Check if any of the selected types match any of the trace's chaos points
            return state.typeFilter.some(selectedType => 
                points.includes(selectedType.toUpperCase())
            );
        });
    }
    
    return traces;
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
// Conversation Timeline Rendering
// ============================================================
function renderConversationEntry(entry, index) {
    const timestamp = formatTimestamp(entry.timestamp_ms);
    
    switch (entry.type) {
        case 'user':
            return `
                <div class="timeline-row user">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="message-bubble">
                            <div class="message-label">USER</div>
                            <div class="message-text">${escapeHtml(entry.content)}</div>
                        </div>
                    </div>
                </div>
            `;
            
        case 'assistant':
            return `
                <div class="timeline-row assistant">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="message-bubble">
                            <div class="message-label">ASSISTANT</div>
                            <div class="message-text">${escapeHtml(entry.content)}</div>
                        </div>
                    </div>
                </div>
            `;
            
        case 'thinking':
            return `
                <div class="timeline-row thinking assistant">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="message-bubble">
                            <div class="message-label">💭 THINKING</div>
                            <div class="message-text">${escapeHtml(truncateText(entry.content, 300))}</div>
                        </div>
                    </div>
                </div>
            `;
            
        case 'tool_call':
            const argsJson = entry.args ? JSON.stringify(entry.args, null, 2) : null;
            return `
                <div class="timeline-row tool_call">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="message-bubble">
                            <div class="message-label">ASSISTANT → TOOL</div>
                            <div class="tool-header">
                                <span class="tool-icon">🔧</span>
                                <span class="tool-name">${escapeHtml(entry.tool_name)}</span>
                            </div>
                            ${argsJson ? `<div class="tool-args"><pre>${escapeHtml(argsJson)}</pre></div>` : ''}
                        </div>
                    </div>
                </div>
            `;
            
        case 'tool_result':
            const resultClass = entry.success === false ? 'error' : '';
            const toolNameDisplay = entry.tool_name && entry.tool_name !== 'unknown' ? entry.tool_name : '(tool)';
            return `
                <div class="timeline-row tool_result ${resultClass}">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="message-bubble">
                            <div class="tool-header">
                                <span class="tool-icon">${entry.success === false ? '❌' : '✓'}</span>
                                <span class="tool-name">${escapeHtml(toolNameDisplay)}</span>
                                ${entry.duration_ms ? `<span class="tool-duration">${formatMs(entry.duration_ms)}</span>` : ''}
                            </div>
                            <div class="tool-result-content">${escapeHtml(entry.result || entry.error || '(no result)')}</div>
                        </div>
                    </div>
                </div>
            `;
            
        case 'chaos':
            let diffHtml = '';
            // Priority 1: Context mutations with added messages
            if (entry.added_messages && Array.isArray(entry.added_messages) && entry.added_messages.length > 0) {
                const addedHtml = entry.added_messages.map(msg => {
                    const content = msg.content || '';
                    return `
                        <div class="context-message added">
                            <span class="msg-role">[${escapeHtml(msg.role)}]</span>

                            <span class="msg-content">${escapeHtml(content)}</span>
                            
                        </div>
                    `;
                }).join('');
                diffHtml = `
                    <div class="chaos-diff context-diff">
                        <div class="diff-header">Injected messages:</div>
                        ${addedHtml}
                    </div>
                `;
            }
            // Priority 2: Context mutations with removed messages
            else if (entry.removed_messages && Array.isArray(entry.removed_messages) && entry.removed_messages.length > 0) {
                const removedHtml = entry.removed_messages.map(msg => {
                    const content = msg.content || '';
                    return `
                        <div class="context-message removed">
                            <span class="msg-role">[${escapeHtml(msg.role)}]</span>
                            <span class="msg-content">${escapeHtml(content)}</span>
                        </div>
                    `;
                }).join('');
                diffHtml = `
                    <div class="chaos-diff context-diff">
                        <div class="diff-header">Removed messages:</div>
                        ${removedHtml}
                    </div>
                `;
            }
            // Priority 3: Tool mutations (original → mutated)
            else if (entry.original && entry.mutated) {
                diffHtml = `
                    <div class="chaos-diff">
                        <span class="diff-line removed">${escapeHtml(entry.original)}</span>
                        <span class="diff-line added">${escapeHtml(entry.mutated)}</span>
                    </div>
                `;
            }
            // Fallback: just show original summary
            else if (entry.original) {
                diffHtml = `<div class="chaos-summary">${escapeHtml(entry.original)}</div>`;
            }
            
            return `
                <div class="timeline-row chaos">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="chaos-banner">
                            <div class="chaos-header">
                                <span class="chaos-icon">⚡</span>
                                <span class="chaos-type">${escapeHtml(entry.fault_type)}</span>
                                ${entry.chaos_fn_name ? `<span class="chaos-fn-name">· ${escapeHtml(entry.chaos_fn_name)}</span>` : ''}
                            </div>
                            <div class="chaos-details">
                                ${entry.chaos_fn_doc ? `<div class="chaos-doc">"${escapeHtml(entry.chaos_fn_doc)}"</div>` : ''}
                                ${entry.target_tool ? `<div>Target: ${escapeHtml(entry.target_tool)}</div>` : ''}
                                ${diffHtml}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
        default:
            return '';
    }
}

function renderConversationTimeline(trace) {
    const conversation = buildConversation(trace);
    
    if (conversation.length === 0) {
        return `
            <div class="empty-state">
                <div class="empty-icon">💬</div>
                <h3 class="empty-title">No conversation captured</h3>
                <p class="empty-text">The agent input/output was not recorded for this run.</p>
            </div>
        `;
    }
    
    return `
        <div class="conversation-timeline">
            ${conversation.map((entry, i) => renderConversationEntry(entry, i)).join('')}
        </div>
    `;
}

function renderSummarySections(trace) {
    const report = trace.report || {};
    const assertions = report.assertion_results || [];
    const faults = extractFaults(trace);
    const passedCount = assertions.filter(a => a.passed).length;
    const failedCount = assertions.filter(a => !a.passed).length;
    
    let html = '<div class="summary-sections">';
    
    // Chaos section first (what was injected)
    html += `
        <div class="summary-section">
            <div class="section-header-row">
                <span class="section-label">Chaos</span>
                <span class="section-count ${faults.length > 0 ? 'has-chaos' : ''}">${faults.length}</span>
            </div>
            ${faults.length > 0 ? `
                <div class="section-items">
                    ${faults.map(f => {
                        const point = f.chaos_point || getChaosPointFallback(f.type);
                        const pointLower = point.toLowerCase();
                        return `
                        <div class="section-item chaos-item">
                            <span class="chaos-category-tag ${pointLower}">${getChaosTypeLabel(pointLower)}</span>
                            <span class="item-text">${escapeHtml(f.type)}${f.target_tool ? ` → ${escapeHtml(f.target_tool)}` : ''}</span>
                        </div>
                    `;}).join('')}
                </div>
            ` : '<div class="section-empty">No chaos injected</div>'}
        </div>
    `;
    
    // Assertions section (verification)
    html += `
        <div class="summary-section">
            <div class="section-header-row">
                <span class="section-label">Assertions</span>
                <span class="section-count">${failedCount > 0 ? `<span class="count-fail">${failedCount}✗</span> <span class="count-pass">${passedCount}✓</span>` : `<span class="count-pass">${passedCount}✓</span>`}</span>
            </div>
            ${assertions.length > 0 ? `
                <div class="section-items">
                    ${assertions.map(a => `
                        <div class="section-item assertion-item ${a.passed ? 'passed' : 'failed'}">
                            <span class="item-icon">${a.passed ? '✓' : '✗'}</span>
                            <span class="item-name">${escapeHtml(a.name)}</span>
                            ${a.message ? `<span class="item-detail"><span class="detail-text">${escapeHtml(a.message)}</span><span class="detail-full">${escapeHtml(a.message)}</span></span>` : ''}
                        </div>
                    `).join('')}
                </div>
            ` : '<div class="section-empty">No assertions defined</div>'}
        </div>
    `;
    
    html += '</div>';
    return html;
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
    const description = trace.description || '';
    
    document.getElementById('modalTitle').innerHTML = `
        ${escapeHtml(trace.name)}
        <span class="outcome-badge ${passed ? 'pass' : 'fail'}">${passed ? 'PASS' : 'FAIL'}</span>
    `;
    // Show trace_id and description together in subtitle - full text, no truncation
    const subtitleText = description 
        ? `${trace.trace_id} · ${description}`
        : trace.trace_id;
    document.getElementById('modalSubtitle').textContent = subtitleText;
    
    document.getElementById('modalBody').innerHTML = 
        renderConversationTimeline(trace) + 
        renderSummarySections(trace);
    
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeScenarioModal() {
    document.getElementById('scenarioModal').classList.add('hidden');
    document.body.style.overflow = '';
    state.selectedTraceId = null;
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
                description: event.data?.description || '',
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
                    trace.fault_count = event.data.fault_count || event.data.chaos_count;
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
            
        case 'tool_use':
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
    
    // Pass/fail filter tabs
    document.querySelectorAll('#filterTabs .filter-tab').forEach(tab => {
        tab.addEventListener('click', () => applyFilter(tab.dataset.filter));
    });
    
    // Chaos type dropdown
    const dropdownTrigger = document.getElementById('chaosTypeDropdownTrigger');
    const dropdown = document.getElementById('chaosTypeDropdown');
    
    dropdownTrigger.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleDropdown();
    });
    
    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!dropdown.contains(e.target) && !dropdownTrigger.contains(e.target)) {
            closeDropdown();
        }
    });
    
    // Handle checkbox changes
    document.querySelectorAll('#chaosTypeDropdown input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            toggleTypeFilter(checkbox.dataset.type);
        });
    });
    
    // Handle chip remove clicks
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('chip-remove')) {
            e.stopPropagation();
            const type = e.target.dataset.type;
            toggleTypeFilter(type);
        }
    });
    
    // Initialize UI
    updateChaosTypeFilterUI();
    
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

// Position detail tooltips on hover
document.addEventListener('mouseover', (e) => {
    const detail = e.target.closest('.item-detail');
    if (detail) {
        const fullEl = detail.querySelector('.detail-full');
        if (fullEl) {
            const rect = detail.getBoundingClientRect();
            fullEl.style.top = `${rect.bottom + 8}px`;
            fullEl.style.left = `${Math.max(10, rect.left - 100)}px`;
        }
    }
});

document.addEventListener('DOMContentLoaded', init);

