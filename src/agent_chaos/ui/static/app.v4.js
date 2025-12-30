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
    viewMode: localStorage.getItem('viewMode') || 'grid', // 'grid' or 'list'
    sortColumn: 'timestamp', // default sort by time
    sortDirection: 'desc', // 'asc' or 'desc'
    groupByTags: localStorage.getItem('groupByTags') === 'true', // Group scenarios by tags
    collapsedGroups: new Set(), // Track collapsed tag groups
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

function renderMarkdown(str) {
    if (str == null) return '';
    try {
        // Configure marked for safety
        marked.setOptions({
            breaks: true,      // Convert \n to <br>
            gfm: true,         // GitHub Flavored Markdown
        });
        return marked.parse(String(str));
    } catch (e) {
        // Fallback to escaped HTML if marked fails
        return escapeHtml(str);
    }
}

// Mapping for chaos type display names
const CHAOS_TYPE_LABELS = {
    'none': 'None',
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
            statsHtml = `<span class="inline-stats"><span class="stat-chaos has-tooltip">⚡${chaosCount}${chaosTooltipHtml}</span><span class="stat-pass has-tooltip">✓${passedCount}${passedTooltipHtml}</span><span class="stat-fail has-tooltip">✗${failedAssertions.length}${failedTooltipHtml}</span></span>`;
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

function renderScenarioListItem(trace) {
    const report = trace.report || {};
    const passed = trace.status === 'success' || report.passed;
    const isRunning = trace.status === 'running';

    let rowClass = 'scenario-list-item';
    if (isRunning) rowClass += ' running';
    else if (passed) rowClass += ' passed';
    else rowClass += ' failed';

    const elapsedS = report.elapsed_s || report.scorecard?.elapsed_s;
    const assertions = report.assertion_results || [];
    const failedAssertions = assertions.filter(a => !a.passed);
    const passedAssertions = assertions.filter(a => a.passed);
    const failedCount = failedAssertions.length;
    const passedCount = passedAssertions.length;
    const chaosCount = trace.fault_count || 0;
    const faults = extractFaults(trace);
    const tools = extractTools(trace);
    const llmCalls = trace.total_calls || 0;
    const toolCalls = tools.length;
    const description = trace.description || '';

    // Get chaos type for compact display
    let chaosTypeLabel = '—';
    if (faults.length > 0) {
        const uniquePoints = new Set(faults.map(f => f.chaos_point || getChaosPointFallback(f.type)));
        if (uniquePoints.size > 1) {
            chaosTypeLabel = 'Multi';
        } else {
            const point = [...uniquePoints][0] || 'unknown';
            chaosTypeLabel = getChaosTypeLabel(point.toLowerCase());
        }
    }

    // Format timestamp (when the run happened)
    const runTime = trace.end_time || trace.start_time;
    const timestamp = runTime ? formatTime(runTime) : '—';

    // Get turn count
    const turnResults = report.turn_results || [];
    const turnCount = turnResults.length || 1;

    // Build tooltips
    const descTooltip = description
        ? `<div class="tooltip"><div class="tooltip-title">Description</div><div class="tooltip-item tooltip-description">${escapeHtml(description)}</div></div>`
        : '';

    const chaosTooltip = faults.length > 0
        ? `<div class="tooltip"><div class="tooltip-title">Chaos Injected</div>${faults.map(f => `<div class="tooltip-item">${escapeHtml(f.type)}${f.target_tool ? ` → ${escapeHtml(f.target_tool)}` : ''}</div>`).join('')}</div>`
        : '';

    const toolsTooltip = tools.length > 0
        ? `<div class="tooltip"><div class="tooltip-title">Tool Calls</div>${tools.map(t => `<div class="tooltip-item">${escapeHtml(t.name)}</div>`).join('')}</div>`
        : '';

    const passTooltip = passedAssertions.length > 0
        ? `<div class="tooltip"><div class="tooltip-title">Passed</div>${passedAssertions.map(a => `<div class="tooltip-item tooltip-pass">✓ ${escapeHtml(a.name)}</div>`).join('')}</div>`
        : '';

    const failTooltip = failedAssertions.length > 0
        ? `<div class="tooltip"><div class="tooltip-title">Failed</div>${failedAssertions.map(a => `<div class="tooltip-item tooltip-fail">✗ ${escapeHtml(a.name)}</div>`).join('')}</div>`
        : '';

    return `
        <div class="${rowClass}" data-trace-id="${trace.trace_id}" data-timestamp="${runTime || ''}">
            <div class="list-name ${description ? 'has-tooltip' : ''}">${escapeHtml(trace.name)}${descTooltip}</div>
            <div class="list-col list-turns">${turnCount}</div>
            <div class="list-col list-chaos-type">${chaosTypeLabel}</div>
            <div class="list-col list-chaos-count ${chaosCount > 0 ? 'has-tooltip' : ''}">${chaosCount > 0 ? `⚡${chaosCount}` : '—'}${chaosTooltip}</div>
            <div class="list-col list-llm">${llmCalls}</div>
            <div class="list-col list-tools ${toolCalls > 0 ? 'has-tooltip' : ''}">${toolCalls}${toolsTooltip}</div>
            <div class="list-col list-pass ${passedCount > 0 ? 'has-tooltip' : ''}">${passedCount > 0 ? `✓${passedCount}` : '—'}${passTooltip}</div>
            <div class="list-col list-fail ${failedCount > 0 ? 'has-tooltip' : ''}">${failedCount > 0 ? `✗${failedCount}` : '—'}${failTooltip}</div>
            <div class="list-col list-duration">${elapsedS ? formatDuration(elapsedS) : '—'}</div>
            <div class="list-col list-timestamp">${timestamp}</div>
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

// ============================================================
// View Toggle (Grid/List)
// ============================================================
function setViewMode(mode) {
    state.viewMode = mode;
    localStorage.setItem('viewMode', mode);
    updateViewToggleUI();
    renderScenarios();
}

function updateViewToggleUI() {
    document.querySelectorAll('#viewToggle .view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === state.viewMode);
    });
    const grid = document.getElementById('scenariosGrid');
    grid.classList.toggle('list-view', state.viewMode === 'list');
}

// ============================================================
// Tag Grouping
// ============================================================
function getTraceTag(trace) {
    // Extract primary tag from trace report (first tag only)
    const report = trace.report || {};
    const tags = report.tags || [];
    return tags.length > 0 ? tags[0] : 'untagged';
}

function groupTracesByTag(traces) {
    // Group traces by their primary tag (first tag only - each trace appears once)
    const groups = {};
    traces.forEach(t => {
        const tag = getTraceTag(t);
        if (!groups[tag]) groups[tag] = [];
        groups[tag].push(t);
    });
    return groups;
}

function toggleGroupByTags() {
    state.groupByTags = !state.groupByTags;
    localStorage.setItem('groupByTags', state.groupByTags);
    updateGroupByTagsUI();
    renderScenarios();
}

function updateGroupByTagsUI() {
    const btn = document.getElementById('groupByTagsBtn');
    if (btn) {
        btn.classList.toggle('active', state.groupByTags);
    }
}

function toggleTagGroup(tag) {
    if (state.collapsedGroups.has(tag)) {
        state.collapsedGroups.delete(tag);
    } else {
        state.collapsedGroups.add(tag);
    }
    renderScenarios();
}

function renderTagGroup(tag, traces) {
    const isCollapsed = state.collapsedGroups.has(tag);

    // Aggregate stats across all scenarios in this group
    let totalChaos = 0;
    let totalAssertionsPassed = 0;
    let totalAssertionsFailed = 0;

    traces.forEach(t => {
        totalChaos += t.fault_count || 0;
        const assertions = t.report?.assertion_results || [];
        assertions.forEach(a => {
            if (a.passed) totalAssertionsPassed++;
            else totalAssertionsFailed++;
        });
    });

    let scenariosHtml;
    if (state.viewMode === 'list') {
        scenariosHtml = traces.map(t => renderScenarioListItem(t)).join('');
    } else {
        scenariosHtml = traces.map(t => renderScenarioCard(t)).join('');
    }

    return `
        <div class="tag-group ${isCollapsed ? 'collapsed' : ''}" data-tag="${escapeHtml(tag)}">
            <div class="tag-group-header" onclick="toggleTagGroup('${escapeHtml(tag).replace(/'/g, "\\'")}')">
                <div class="tag-group-info">
                    <span class="tag-group-chevron">${isCollapsed ? '▶' : '▼'}</span>
                    <span class="tag-group-name">${escapeHtml(tag)}</span>
                    <span class="tag-group-count">${traces.length}</span>
                </div>
                <div class="tag-group-stats">
                    ${totalChaos > 0 ? `<span class="tag-stat chaos">⚡${totalChaos}</span>` : ''}
                    ${totalAssertionsPassed > 0 ? `<span class="tag-stat pass">✓${totalAssertionsPassed}</span>` : ''}
                    ${totalAssertionsFailed > 0 ? `<span class="tag-stat fail">✗${totalAssertionsFailed}</span>` : ''}
                </div>
            </div>
            <div class="tag-group-content ${state.viewMode === 'list' ? 'list-content' : 'grid-content'}">
                ${scenariosHtml}
            </div>
        </div>
    `;
}

// ============================================================
// Sorting
// ============================================================
function toggleSort(column) {
    if (state.sortColumn === column) {
        // Toggle direction
        state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        state.sortColumn = column;
        state.sortDirection = 'desc'; // Default to descending for new column
    }
    renderScenarios();
}

function getSortValue(trace, column) {
    const report = trace.report || {};
    const assertions = report.assertion_results || [];

    switch (column) {
        case 'name':
            return trace.name?.toLowerCase() || '';
        case 'turns':
            return (report.turn_results || []).length || 1;
        case 'type':
            const faults = extractFaults(trace);
            if (faults.length === 0) return '';
            const points = new Set(faults.map(f => f.chaos_point || getChaosPointFallback(f.type)));
            return [...points][0] || '';
        case 'chaos':
            return trace.fault_count || 0;
        case 'llm':
            return trace.total_calls || 0;
        case 'tools':
            return extractTools(trace).length;
        case 'pass':
            return assertions.filter(a => a.passed).length;
        case 'fail':
            return assertions.filter(a => !a.passed).length;
        case 'duration':
            return report.elapsed_s || report.scorecard?.elapsed_s || 0;
        case 'timestamp':
            return trace.end_time || trace.start_time || '';
        default:
            return '';
    }
}

function sortTraces(traces) {
    return traces.sort((a, b) => {
        const valA = getSortValue(a, state.sortColumn);
        const valB = getSortValue(b, state.sortColumn);

        let comparison = 0;
        if (typeof valA === 'number' && typeof valB === 'number') {
            comparison = valA - valB;
        } else {
            comparison = String(valA).localeCompare(String(valB));
        }

        return state.sortDirection === 'asc' ? comparison : -comparison;
    });
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

            // Check if any of the selected types match
            return state.typeFilter.some(selectedType => {
                // Special case: "none" matches traces with no chaos
                if (selectedType === 'none') {
                    return faults.length === 0;
                }
                return points.includes(selectedType.toUpperCase());
            });
        });
    }

    return traces;
}

function renderScenarios() {
    const grid = document.getElementById('scenariosGrid');
    let traces = getFilteredTraces();

    // Apply view mode class
    grid.classList.toggle('list-view', state.viewMode === 'list');
    grid.classList.toggle('grouped-view', state.groupByTags);

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

    // Sort traces
    if (state.viewMode === 'list') {
        traces = sortTraces(traces);
    } else {
        // Grid view: default sort by time descending
        traces.sort((a, b) => {
            const timeA = a.end_time || a.start_time || '';
            const timeB = b.end_time || b.start_time || '';
            return timeB.localeCompare(timeA);
        });
    }

    // Helper for sort indicator
    const sortIcon = (col) => {
        if (state.sortColumn !== col) return '<span class="sort-icon">⇅</span>';
        return state.sortDirection === 'asc'
            ? '<span class="sort-icon active">↑</span>'
            : '<span class="sort-icon active">↓</span>';
    };

    // Render grouped by tags if enabled
    if (state.groupByTags) {
        const groups = groupTracesByTag(traces);
        const tags = Object.keys(groups).sort();

        grid.innerHTML = tags.map(tag => renderTagGroup(tag, groups[tag])).join('');

        // Add click handlers for scenario cards/items within groups
        grid.querySelectorAll('.scenario-card').forEach(card => {
            card.addEventListener('click', () => openScenarioModal(card.dataset.traceId));
        });
        grid.querySelectorAll('.scenario-list-item').forEach(item => {
            item.addEventListener('click', () => openScenarioModal(item.dataset.traceId));
        });
        return;
    }

    // Render based on view mode (flat, non-grouped)
    if (state.viewMode === 'list') {
        grid.innerHTML = `
            <div class="list-header">
                <div class="list-name sortable" data-sort="name">Scenario ${sortIcon('name')}</div>
                <div class="list-col list-turns sortable" data-sort="turns">Turns ${sortIcon('turns')}</div>
                <div class="list-col list-chaos-type sortable" data-sort="type">Type ${sortIcon('type')}</div>
                <div class="list-col list-chaos-count sortable" data-sort="chaos">Chaos ${sortIcon('chaos')}</div>
                <div class="list-col list-llm sortable" data-sort="llm">LLM ${sortIcon('llm')}</div>
                <div class="list-col list-tools sortable" data-sort="tools">Tools ${sortIcon('tools')}</div>
                <div class="list-col list-pass sortable" data-sort="pass">Pass ${sortIcon('pass')}</div>
                <div class="list-col list-fail sortable" data-sort="fail">Fail ${sortIcon('fail')}</div>
                <div class="list-col list-duration sortable" data-sort="duration">Duration ${sortIcon('duration')}</div>
                <div class="list-col list-timestamp sortable" data-sort="timestamp">Time ${sortIcon('timestamp')}</div>
            </div>
            ${traces.map(t => renderScenarioListItem(t)).join('')}
        `;
        // Add click handlers for sorting
        grid.querySelectorAll('.list-header .sortable').forEach(header => {
            header.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleSort(header.dataset.sort);
            });
        });
        grid.querySelectorAll('.scenario-list-item').forEach(item => {
            item.addEventListener('click', () => openScenarioModal(item.dataset.traceId));
        });
    } else {
        grid.innerHTML = traces.map(t => renderScenarioCard(t)).join('');
        grid.querySelectorAll('.scenario-card').forEach(card => {
            card.addEventListener('click', () => openScenarioModal(card.dataset.traceId));
        });
    }
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
function renderTurnSeparator(entry) {
    const turnNum = entry.turn_number || 1;
    const isStart = entry.type === 'turn_start';

    if (isStart) {
        const inputType = entry.input_type === 'dynamic' ? '<span class="turn-dynamic">λ dynamic</span>' : '';
        return `
            <div class="timeline-row turn-separator turn-start">
                <div class="time-gutter"></div>
                <div class="timeline-content">
                    <div class="turn-header">
                        <span class="turn-label">TURN ${turnNum}</span>
                        ${inputType}
                    </div>
                </div>
            </div>
        `;
    } else {
        // turn_end
        const status = entry.success ? '✓' : '✗';
        const statusClass = entry.success ? 'success' : 'failed';
        const duration = entry.duration_s ? formatDuration(entry.duration_s) : '';
        const llmCalls = entry.llm_calls ? `${entry.llm_calls} LLM` : '';

        return `
            <div class="timeline-row turn-separator turn-end">
                <div class="time-gutter"></div>
                <div class="timeline-content">
                    <div class="turn-footer ${statusClass}">
                        <span class="turn-status">${status}</span>
                        <span class="turn-stats">${duration}${llmCalls ? ' · ' + llmCalls : ''}</span>
                        ${entry.error ? `<span class="turn-error">${escapeHtml(entry.error)}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    }
}

function renderConversationEntry(entry, index) {
    const timestamp = formatTimestamp(entry.timestamp_ms);

    switch (entry.type) {
        case 'turn_start':
        case 'turn_end':
            return renderTurnSeparator(entry);

        case 'between_turn_chaos':
            return `
                <div class="timeline-row between-turns-chaos">
                    <div class="time-gutter"></div>
                    <div class="timeline-content">
                        <div class="between-turns-banner">
                            <div class="between-turns-header">
                                <span class="chaos-icon">⚡</span>
                                <span>BETWEEN TURNS ${entry.after_turn} → ${entry.before_turn}</span>
                            </div>
                            <div class="chaos-type">${escapeHtml(entry.chaos_type || 'history_mutate')}</div>
                        </div>
                    </div>
                </div>
            `;

        case 'user':
            const dynamicBadge = entry.is_dynamic ? '<span class="dynamic-badge">λ</span>' : '';
            const turnLabel = entry.turn_number ? `<span class="turn-indicator">T${entry.turn_number}</span>` : '';
            return `
                <div class="timeline-row user">
                    <div class="time-gutter">
                        <span class="time-label">${timestamp}</span>
                    </div>
                    <div class="timeline-content">
                        <div class="message-bubble">
                            <div class="message-label">USER ${dynamicBadge}${turnLabel}</div>
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
                            <div class="message-text markdown-content">${renderMarkdown(entry.content)}</div>
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
    const report = trace.report || {};
    const turnResults = report.turn_results || [];

    if (conversation.length === 0) {
        return `
            <div class="empty-state">
                <div class="empty-icon">💬</div>
                <h3 class="empty-title">No conversation captured</h3>
                <p class="empty-text">The agent input/output was not recorded for this run.</p>
            </div>
        `;
    }

    // Render turn-based timeline
    return renderMultiTurnTimeline(conversation, turnResults);
}

function renderMultiTurnTimeline(conversation, turnResults) {
    // If no turn results yet (e.g., live streaming), show flat timeline
    if (!turnResults || turnResults.length === 0) {
        return `
            <div class="conversation-timeline">
                ${conversation.map((entry, i) => renderConversationEntry(entry, i)).join('')}
            </div>
        `;
    }

    // Group conversation entries by turn_number
    const turnGroups = {};
    const betweenTurnsChaos = []; // Chaos that happens between turns

    conversation.forEach(entry => {
        const turnNum = entry.turn_number || 0;

        // Handle between_turn_chaos specially
        if (entry.type === 'between_turn_chaos') {
            betweenTurnsChaos.push(entry);
            return;
        }

        // Skip turn_start and turn_end markers - we render our own headers
        if (entry.type === 'turn_start' || entry.type === 'turn_end') {
            return;
        }

        if (!turnGroups[turnNum]) turnGroups[turnNum] = [];
        turnGroups[turnNum].push(entry);
    });

    let html = '<div class="turns-container">';

    turnResults.forEach((turnResult, index) => {
        const turnNum = turnResult.turn_number;
        const entries = turnGroups[turnNum] || [];
        const hasChaos = entries.some(e => e.type === 'chaos') || (turnResult.chaos && turnResult.chaos.length > 0);
        const turnAssertions = turnResult.assertion_results || [];
        const turnChaos = turnResult.chaos || [];

        // Check for between-turns chaos before this turn
        const betweenChaos = betweenTurnsChaos.filter(c =>
            c.after_turn === turnNum - 1 && c.before_turn === turnNum
        );

        // Render between-turns chaos if any
        if (betweenChaos.length > 0 && index > 0) {
            html += `
                <div class="between-turns">
                    ${betweenChaos.map(c => `
                        <div class="between-turns-banner">
                            <span class="between-turns-label">Between Turns</span>
                            <span class="between-turns-chaos">
                                <span>⚡</span>
                                <span>${escapeHtml(c.chaos_type || 'context_mutate')}${c.chaos_fn_name ? ` · ${c.chaos_fn_name}` : ''}</span>
                            </span>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Count tool calls for this turn
        const toolCalls = entries.filter(e => e.type === 'tool_call').length;

        // Build header stats
        const chaosCount = turnChaos.length;
        const passedCount = turnAssertions.filter(a => a.passed).length;
        const failedCount = turnAssertions.filter(a => !a.passed).length;

        // Render the turn section
        html += `
            <div class="turn" data-turn="${turnNum}">
                <div class="turn-header" onclick="toggleTurn(this)">
                    <div class="turn-label">
                        TURN ${turnNum}
                        <span class="turn-status ${turnResult.success ? 'success' : 'error'}">${turnResult.success ? '✓' : '✗'}</span>
                        ${chaosCount > 0 ? `<span class="turn-chaos-count">⚡${chaosCount}</span>` : ''}
                        ${passedCount > 0 ? `<span class="turn-pass-count">✓${passedCount}</span>` : ''}
                        ${failedCount > 0 ? `<span class="turn-fail-count">✗${failedCount}</span>` : ''}
                        ${turnResult.is_dynamic ? '<span class="dynamic-indicator">λ</span>' : ''}
                    </div>
                    <div class="turn-stats">
                        <span>${turnResult.llm_calls || 0} LLM</span>
                        <span>${toolCalls} tools</span>
                        <span>${formatDuration(turnResult.duration_s)}</span>
                        <span class="turn-chevron">▼</span>
                    </div>
                </div>
                <div class="turn-content">
                    <div class="conversation-timeline">
                        ${entries.map((entry, i) => renderConversationEntry(entry, i)).join('')}
                    </div>
                    ${turnResult.error ? `<div class="turn-error-banner">${escapeHtml(turnResult.error)}</div>` : ''}
                    ${renderTurnFooter(turnChaos, turnAssertions)}
                </div>
            </div>
        `;
    });

    html += '</div>';
    return html;
}

/**
 * Build tooltip HTML for a chaos item.
 */
function buildChaosTooltip(c) {
    const rows = [];

    if (c.chaos_type) {
        rows.push(`<div class="tip-row"><span class="tip-label">Type:</span> <span class="tip-value">${escapeHtml(c.chaos_type)}</span></div>`);
    }
    if (c.target_tool) {
        rows.push(`<div class="tip-row"><span class="tip-label">Target:</span> <span class="tip-value mono">${escapeHtml(c.target_tool)}</span></div>`);
    }
    if (c.message) {
        rows.push(`<div class="tip-row"><span class="tip-label">Message:</span> <span class="tip-value">${escapeHtml(c.message)}</span></div>`);
    }
    if (c.chaos_fn_name) {
        rows.push(`<div class="tip-row"><span class="tip-label">Function:</span> <span class="tip-value mono">${escapeHtml(c.chaos_fn_name)}</span></div>`);
    }
    if (c.chaos_fn_doc) {
        rows.push(`<div class="tip-row"><span class="tip-label">Doc:</span> <span class="tip-value italic">${escapeHtml(c.chaos_fn_doc)}</span></div>`);
    }
    if (c.on_turn != null) {
        rows.push(`<div class="tip-row"><span class="tip-label">On Turn:</span> <span class="tip-value">${c.on_turn}</span></div>`);
    }
    if (c.after_calls != null) {
        rows.push(`<div class="tip-row"><span class="tip-label">After Calls:</span> <span class="tip-value">${c.after_calls}</span></div>`);
    }

    if (rows.length === 0) return '';

    return `
        <div class="panel-tooltip">
            <div class="tip-header">Chaos Details</div>
            ${rows.join('')}
        </div>
    `;
}

/**
 * Build tooltip HTML for an assertion item.
 */
function buildAssertionTooltip(a) {
    const rows = [];

    rows.push(`<div class="tip-row"><span class="tip-label">Name:</span> <span class="tip-value">${escapeHtml(a.name)}</span></div>`);
    rows.push(`<div class="tip-row"><span class="tip-label">Status:</span> <span class="tip-value ${a.passed ? 'tip-pass' : 'tip-fail'}">${a.passed ? 'PASSED' : 'FAILED'}</span></div>`);

    if (a.measured != null) {
        const measuredStr = typeof a.measured === 'number' ? a.measured.toFixed(4) : String(a.measured);
        rows.push(`<div class="tip-row"><span class="tip-label">Measured:</span> <span class="tip-value mono">${escapeHtml(measuredStr)}</span></div>`);
    }
    if (a.expected != null) {
        rows.push(`<div class="tip-row"><span class="tip-label">Expected:</span> <span class="tip-value mono">${escapeHtml(String(a.expected))}</span></div>`);
    }
    if (a.message) {
        rows.push(`<div class="tip-row tip-message"><span class="tip-label">Message:</span> <span class="tip-value">${escapeHtml(a.message)}</span></div>`);
    }

    return `
        <div class="panel-tooltip">
            <div class="tip-header">Assertion Details</div>
            ${rows.join('')}
        </div>
    `;
}

/**
 * Shared renderer for chaos/assertions panels.
 * Used by both turn-level footers and scenario-level panels.
 *
 * @param {Array} chaos - Array of chaos config objects
 * @param {Array} assertions - Array of assertion result objects
 * @param {Object} options - Rendering options
 * @param {string} options.scope - 'turn' or 'scenario' (affects labels and styling)
 * @param {string} options.wrapperClass - Additional CSS class for the wrapper
 */
function renderChaosAssertionsPanel(chaos, assertions, options = {}) {
    const scope = options.scope || 'turn';
    const wrapperClass = options.wrapperClass || '';
    const labelPrefix = scope === 'turn' ? 'Turn ' : 'Scenario ';

    const chaosCount = (chaos || []).length;
    const passedCount = (assertions || []).filter(a => a.passed).length;
    const failedCount = (assertions || []).filter(a => !a.passed).length;

    const chaosItemsHtml = chaosCount > 0
        ? (chaos || []).map(c => {
            const turnInfo = scope === 'scenario' && c.on_turn ? ` (turn ${c.on_turn})` : '';
            const allTurns = scope === 'scenario' && !c.on_turn ? ' (all turns)' : '';
            const tooltip = buildChaosTooltip(c);
            return `
                <div class="panel-item chaos-item has-panel-tooltip">
                    <span class="item-icon chaos-icon">⚡</span>
                    <span class="item-name">${escapeHtml(c.chaos_type || 'chaos')}</span>
                    ${c.target_tool ? `<span class="item-target">→ ${escapeHtml(c.target_tool)}</span>` : ''}
                    <span class="item-scope">${turnInfo}${allTurns}</span>
                    ${c.chaos_fn_doc ? `<div class="item-doc">"${escapeHtml(truncateText(c.chaos_fn_doc, 60))}"</div>` : ''}
                    ${tooltip}
                </div>
            `;
        }).join('')
        : `<div class="panel-empty">No chaos</div>`;

    const assertionItemsHtml = (assertions || []).length > 0
        ? (assertions || []).map(a => {
            const tooltip = buildAssertionTooltip(a);
            return `
                <div class="panel-item assertion-item ${a.passed ? 'passed' : 'failed'} has-panel-tooltip">
                    <span class="item-icon">${a.passed ? '✓' : '✗'}</span>
                    <span class="item-name">${escapeHtml(a.name)}</span>
                    ${a.measured != null ? `<span class="item-score">(${typeof a.measured === 'number' ? a.measured.toFixed(2) : a.measured})</span>` : ''}
                    ${a.message ? `<div class="item-message">${escapeHtml(truncateText(a.message, 60))}</div>` : ''}
                    ${tooltip}
                </div>
            `;
        }).join('')
        : `<div class="panel-empty">No assertions</div>`;

    return `
        <div class="chaos-assertions-panel ${wrapperClass}" data-scope="${scope}">
            <div class="panel-col chaos-col">
                <div class="panel-header">
                    <span class="panel-label">${labelPrefix}Chaos</span>
                    <span class="panel-count ${chaosCount > 0 ? 'has-items' : ''}">${chaosCount}</span>
                </div>
                <div class="panel-items">${chaosItemsHtml}</div>
            </div>
            <div class="panel-col assertions-col">
                <div class="panel-header">
                    <span class="panel-label">${labelPrefix}Assertions</span>
                    <span class="panel-count">${failedCount > 0 ? `<span class="count-fail">${failedCount}✗</span> <span class="count-pass">${passedCount}✓</span>` : `<span class="count-pass">${passedCount}✓</span>`}</span>
                </div>
                <div class="panel-items">${assertionItemsHtml}</div>
            </div>
        </div>
    `;
}

function renderTurnFooter(chaos, assertions) {
    // Only render footer if there's chaos or assertions
    if ((!chaos || chaos.length === 0) && (!assertions || assertions.length === 0)) {
        return '';
    }

    return renderChaosAssertionsPanel(chaos, assertions, {
        scope: 'turn',
        wrapperClass: 'turn-footer'
    });
}

function toggleTurn(header) {
    const turn = header.parentElement;
    turn.classList.toggle('collapsed');
}

function renderSummarySections(trace) {
    const report = trace.report || {};
    const scorecard = report.scorecard || {};

    // Scenario-level assertions (turn-level are now in turn_results)
    const scenarioAssertions = report.assertion_results || [];

    // Scenario-level chaos config from scorecard
    const scenarioChaos = scorecard.scenario_chaos || [];

    // Check if there's any scenario-level content to show
    const hasScenarioContent = scenarioChaos.length > 0 || scenarioAssertions.length > 0;

    if (!hasScenarioContent) {
        return ''; // Don't render empty scenario section
    }

    return `
        <div class="scenario-panel-wrapper">
            <div class="scenario-panel-header">SCENARIO</div>
            ${renderChaosAssertionsPanel(scenarioChaos, scenarioAssertions, {
                scope: 'scenario',
                wrapperClass: 'scenario-panel'
            })}
        </div>
    `;
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

    // View toggle (grid/list)
    document.querySelectorAll('#viewToggle .view-btn').forEach(btn => {
        btn.addEventListener('click', () => setViewMode(btn.dataset.view));
    });
    updateViewToggleUI();

    // Group by tags toggle
    const groupByTagsBtn = document.getElementById('groupByTagsBtn');
    if (groupByTagsBtn) {
        groupByTagsBtn.addEventListener('click', toggleGroupByTags);
    }
    updateGroupByTagsUI();

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

    // Position list view tooltips
    const hasTooltip = e.target.closest('.scenario-list-item .has-tooltip');
    if (hasTooltip) {
        const tooltip = hasTooltip.querySelector('.tooltip');
        if (tooltip) {
            const rect = hasTooltip.getBoundingClientRect();
            // Position above the element
            tooltip.style.bottom = 'auto';
            tooltip.style.top = `${rect.top - 8}px`;
            tooltip.style.transform = 'translateY(-100%)';
            // Keep within screen bounds horizontally
            tooltip.style.left = `${Math.max(12, rect.left)}px`;
        }
    }

    // Position scenario card tooltips in grouped view (to avoid overflow clipping)
    const groupedCard = e.target.closest('.grouped-view .scenario-card .has-tooltip');
    if (groupedCard) {
        const tooltip = groupedCard.querySelector('.tooltip');
        if (tooltip) {
            positionCardTooltip(groupedCard, tooltip);
        }
    }
});

// Reset tooltip styles on mouseleave for grouped view cards
document.addEventListener('mouseleave', (e) => {
    const groupedCard = e.target.closest('.grouped-view .scenario-card .has-tooltip');
    if (groupedCard) {
        const tooltip = groupedCard.querySelector('.tooltip');
        if (tooltip) {
            // Reset inline styles so CSS can control visibility
            tooltip.style.position = '';
            tooltip.style.top = '';
            tooltip.style.left = '';
            tooltip.style.bottom = '';
            tooltip.style.transform = '';
            tooltip.style.opacity = '';
            tooltip.style.visibility = '';
        }
    }
}, true);

// Position card tooltip with fixed positioning to avoid overflow clipping
function positionCardTooltip(element, tooltip) {
    const rect = element.getBoundingClientRect();
    const padding = 12;

    // Use fixed positioning and reset CSS-based position values
    tooltip.style.position = 'fixed';
    tooltip.style.bottom = 'auto';
    tooltip.style.transform = 'none';

    // Force visibility so we can measure
    tooltip.style.opacity = '1';
    tooltip.style.visibility = 'visible';

    const tooltipHeight = tooltip.offsetHeight || 100;
    const tooltipWidth = tooltip.offsetWidth || 280;

    // Check space below vs above
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;

    if (spaceBelow >= tooltipHeight + padding) {
        // Position below
        tooltip.style.top = `${rect.bottom + 8}px`;
    } else if (spaceAbove >= tooltipHeight + padding) {
        // Position above
        tooltip.style.top = `${rect.top - tooltipHeight - 8}px`;
    } else {
        // Default below, may need scroll
        tooltip.style.top = `${rect.bottom + 8}px`;
    }

    // Horizontal positioning
    let left = rect.left;
    if (left + tooltipWidth > window.innerWidth - padding) {
        left = window.innerWidth - tooltipWidth - padding;
    }
    tooltip.style.left = `${Math.max(padding, left)}px`;
}

// ============================================================
// Panel Tooltip Positioning
// ============================================================
function positionPanelTooltip(item, tooltip) {
    const itemRect = item.getBoundingClientRect();
    const tooltipHeight = tooltip.offsetHeight || 150; // estimate if not yet visible
    const padding = 12;

    // Check space above and below
    const spaceAbove = itemRect.top;
    const spaceBelow = window.innerHeight - itemRect.bottom;

    // Reset classes
    tooltip.classList.remove('above', 'below');

    if (spaceAbove >= tooltipHeight + padding) {
        // Position above
        tooltip.style.top = `${itemRect.top - tooltipHeight - 8}px`;
        tooltip.classList.add('above');
    } else if (spaceBelow >= tooltipHeight + padding) {
        // Position below
        tooltip.style.top = `${itemRect.bottom + 8}px`;
        tooltip.classList.add('below');
    } else {
        // Default to above, will scroll if needed
        tooltip.style.top = `${Math.max(padding, itemRect.top - tooltipHeight - 8)}px`;
        tooltip.classList.add('above');
    }

    // Horizontal positioning - align to left of item, but keep in viewport
    let left = itemRect.left;
    const tooltipWidth = tooltip.offsetWidth || 280;
    if (left + tooltipWidth > window.innerWidth - padding) {
        left = window.innerWidth - tooltipWidth - padding;
    }
    tooltip.style.left = `${Math.max(padding, left)}px`;
}

document.addEventListener('mouseenter', (e) => {
    const item = e.target.closest('.has-panel-tooltip');
    if (item) {
        const tooltip = item.querySelector('.panel-tooltip');
        if (tooltip) {
            positionPanelTooltip(item, tooltip);
            tooltip.classList.add('visible');
        }
    }
}, true);

document.addEventListener('mouseleave', (e) => {
    const item = e.target.closest('.has-panel-tooltip');
    if (item) {
        const tooltip = item.querySelector('.panel-tooltip');
        if (tooltip) {
            tooltip.classList.remove('visible');
        }
    }
}, true);

document.addEventListener('DOMContentLoaded', init);

