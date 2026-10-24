/**
 * app/static/js/app.js
 * --------------------
 * Single-page application logic.
 * Handles view routing, API calls, and DOM rendering.
 * No framework dependencies.
 */

const API = {
    async get(path) {
        const res = await fetch(path);
        return res.json();
    },
    async post(path, body) {
        const res = await fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        return res.json();
    },
    async delete(path) {
        const res = await fetch(path, { method: "DELETE" });
        return res.json();
    },
};

// ── VIEW ROUTING ──────────────────────────────────────────────

document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => {
        const view = btn.dataset.view;
        document.querySelectorAll(".nav-item").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(`view-${view}`).classList.add("active");
        if (view === "browse") loadBrowse();
    });
});

// ── API STATUS ────────────────────────────────────────────────

async function checkApiStatus() {
    const el = document.getElementById("api-status");
    try {
        const data = await API.get("/health");
        if (data.status === "ok") {
            el.classList.add("ok");
            el.querySelector(".status-text").textContent = "API online";
        } else {
            throw new Error();
        }
    } catch {
        el.classList.add("err");
        el.querySelector(".status-text").textContent = "API offline";
    }
}

checkApiStatus();

// ── TOAST ─────────────────────────────────────────────────────

let toastTimer = null;
function showToast(message) {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.classList.remove("hidden");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.add("hidden"), 3000);
}

// ── MODE TOGGLE (search view) ─────────────────────────────────

document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        const mode = btn.dataset.mode;
        document.getElementById("panel-natural").classList.toggle("hidden", mode !== "natural");
        document.getElementById("panel-structured").classList.toggle("hidden", mode !== "structured");
        document.getElementById("search-results").hidden = true;
    });
});

// ── SPATIAL TYPE TOGGLE ───────────────────────────────────────

document.getElementById("s-spatial-type").addEventListener("change", e => {
    document.getElementById("bbox-fields").classList.toggle("hidden", e.target.value !== "bbox");
    document.getElementById("point-fields").classList.toggle("hidden", e.target.value !== "point");
});

// ── NATURAL LANGUAGE SEARCH ───────────────────────────────────

document.getElementById("nl-submit").addEventListener("click", runNaturalSearch);
document.getElementById("nl-input").addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runNaturalSearch();
});

async function runNaturalSearch() {
    const query = document.getElementById("nl-input").value.trim();
    if (!query) return;

    const btn = document.getElementById("nl-submit");
    setLoading(btn, true);
    showResults([], true);

    try {
        const data = await API.post("/query/natural", { query, limit: 50 });

        if (data.error) {
            showToast(`Error: ${data.error}`);
            return;
        }

        // Show extracted filters
        const filtersEl = document.getElementById("extracted-filters");
        const filtersJson = document.getElementById("filters-json");
        if (data.filters && Object.keys(data.filters).length > 0) {
            filtersJson.textContent = JSON.stringify(data.filters);
            filtersEl.hidden = false;
        } else {
            filtersEl.hidden = true;
        }

        showResults(data.datasets || []);

    } catch (err) {
        showToast("Request failed. Is the API running?");
    } finally {
        setLoading(btn, false);
    }
}

// ── STRUCTURED SEARCH ─────────────────────────────────────────

document.getElementById("s-submit").addEventListener("click", runStructuredSearch);

async function runStructuredSearch() {
    const body = {};

    const fmt = document.getElementById("s-format").value;
    if (fmt) body.format = fmt;

    const variable = document.getElementById("s-variable").value.trim();
    if (variable) body.variable = variable;

    const tStart = document.getElementById("s-time-start").value;
    const tEnd   = document.getElementById("s-time-end").value;
    if (tStart && tEnd) body.temporal = { start: tStart, end: tEnd };

    const spatialType = document.getElementById("s-spatial-type").value;
    if (spatialType === "bbox") {
        body.spatial = {
            type:    "bbox",
            min_lat: parseFloat(document.getElementById("s-min-lat").value),
            max_lat: parseFloat(document.getElementById("s-max-lat").value),
            min_lon: parseFloat(document.getElementById("s-min-lon").value),
            max_lon: parseFloat(document.getElementById("s-max-lon").value),
        };
    } else if (spatialType === "point") {
        body.spatial = {
            type: "point",
            lat:  parseFloat(document.getElementById("s-lat").value),
            lon:  parseFloat(document.getElementById("s-lon").value),
        };
    }

    const btn = document.getElementById("s-submit");
    setLoading(btn, true);
    showResults([], true);

    try {
        const data = await API.post("/query/structured", { ...body, limit: 50 });
        if (data.error) { showToast(`Error: ${data.error}`); return; }
        document.getElementById("extracted-filters").hidden = true;
        showResults(data.datasets || []);
    } catch {
        showToast("Request failed.");
    } finally {
        setLoading(btn, false);
    }
}

// ── RESULTS RENDERING ─────────────────────────────────────────

function showResults(datasets, loading = false) {
    const area  = document.getElementById("search-results");
    const list  = document.getElementById("search-dataset-list");
    const count = document.getElementById("results-count");
    area.hidden = false;

    if (loading) {
        list.innerHTML = `<div class="loading-row"><span class="spinner"></span> Searching…</div>`;
        count.textContent = "";
        return;
    }

    count.textContent = `${datasets.length} result${datasets.length !== 1 ? "s" : ""}`;
    list.innerHTML = datasets.length === 0
        ? `<div class="empty-state"><span class="empty-icon">◈</span>No datasets matched your query.</div>`
        : datasets.map(renderCard).join("");

    list.querySelectorAll(".dataset-card").forEach(card => {
        card.addEventListener("click", () => openDetail(card.dataset.id));
    });
}

function renderCard(ds) {
    const fmt    = (ds.format || "").toLowerCase();
    const source = ds.source || ds.dataset_id || "Unknown source";
    const name   = source.split(/[\\/]/).pop();

    const sp  = ds.spatial || {};
    const tp  = ds.temporal || {};
    const coords = sp.lat_min != null
        ? `${sp.lat_min.toFixed(2)}°, ${sp.lon_min.toFixed(2)}° → ${sp.lat_max.toFixed(2)}°, ${sp.lon_max.toFixed(2)}°`
        : "No spatial info";

    const timeRange = tp.start
        ? `${tp.start.slice(0, 10)} → ${(tp.end || "").slice(0, 10)}`
        : "";

    return `
    <div class="dataset-card" data-id="${ds.dataset_id}">
        <div class="card-main">
            <div class="card-title" title="${source}">${name}</div>
            <div class="card-meta">
                <span class="badge ${fmt}">${fmt || "unknown"}</span>
                ${timeRange ? `<span class="badge">${timeRange}</span>` : ""}
                ${(ds.variables || []).slice(0, 3).map(v =>
                    `<span class="badge">${v.name}</span>`
                ).join("")}
            </div>
            <div class="card-coords">${coords}</div>
        </div>
        <div class="card-actions">
            <button class="btn-danger delete-btn" data-id="${ds.dataset_id}" title="Delete dataset">✕</button>
        </div>
    </div>`;
}

// Delegate delete clicks on result cards
document.getElementById("search-dataset-list").addEventListener("click", e => {
    const btn = e.target.closest(".delete-btn");
    if (btn) { e.stopPropagation(); confirmDelete(btn.dataset.id, "search"); }
});

// ── EXTRACT VIEW ──────────────────────────────────────────────

document.getElementById("extract-submit").addEventListener("click", async () => {
    const source = document.getElementById("extract-source").value.trim();
    const format = document.getElementById("extract-format").value;
    const result = document.getElementById("extract-result");

    if (!source) { showToast("Please enter a source path or URL."); return; }

    const btn = document.getElementById("extract-submit");
    setLoading(btn, true);
    result.className = "extract-result hidden";

    try {
        const data = await API.post("/datasets/extract", { source, format });

        result.classList.remove("hidden");
        if (data.error) {
            result.classList.add("error");
            result.textContent = `Error: ${data.error}`;
        } else {
            result.classList.add("success");
            result.textContent = data.message || "Extracted successfully.";
        }
    } catch {
        result.classList.remove("hidden");
        result.classList.add("error");
        result.textContent = "Request failed. Is the API running?";
    } finally {
        setLoading(btn, false);
    }
});

// ── BROWSE VIEW ───────────────────────────────────────────────

async function loadBrowse() {
    const list = document.getElementById("browse-dataset-list");
    const fmt  = document.getElementById("browse-format-filter").value;

    list.innerHTML = `<div class="loading-row"><span class="spinner"></span> Loading…</div>`;

    try {
        const url  = fmt ? `/datasets/?format=${fmt}` : "/datasets/";
        const data = await API.get(url);
        const datasets = data.datasets || [];

        list.innerHTML = datasets.length === 0
            ? `<div class="empty-state"><span class="empty-icon">▦</span>No datasets in catalog.</div>`
            : datasets.map(renderCard).join("");

        list.querySelectorAll(".dataset-card").forEach(card => {
            card.addEventListener("click", () => openDetail(card.dataset.id));
        });
    } catch {
        list.innerHTML = `<div class="empty-state">Failed to load datasets.</div>`;
    }
}

document.getElementById("browse-refresh").addEventListener("click", loadBrowse);
document.getElementById("browse-format-filter").addEventListener("change", loadBrowse);

// ── DETAIL PANEL ──────────────────────────────────────────────

async function openDetail(datasetId) {
    const overlay = document.getElementById("detail-overlay");
    const content = document.getElementById("detail-content");

    overlay.classList.remove("hidden");
    content.innerHTML = `<div class="loading-row"><span class="spinner"></span> Loading…</div>`;

    try {
        const ds = await API.get(`/datasets/${datasetId}`);
        if (ds.error) { content.innerHTML = `<p style="color:var(--danger)">${ds.error}</p>`; return; }
        content.innerHTML = renderDetail(ds);

        content.querySelector(".detail-delete-btn")?.addEventListener("click", () => {
            confirmDelete(datasetId, "detail");
        });
    } catch {
        content.innerHTML = `<p style="color:var(--danger)">Failed to load dataset.</p>`;
    }
}

function renderVariables(variables) {
    if (!variables || variables.length === 0) return "";

    const rows = variables.map(v => {
        const s = v.statistics;
        const statsHtml = s && s.count > 0
            ? `<span class="stat">μ ${fmt(s.mean)}</span>
               <span class="stat">σ ${fmt(s.std)}</span>
               <span class="stat">↓ ${fmt(s.min)}</span>
               <span class="stat">↑ ${fmt(s.max)}</span>
               <span class="stat nan">${`${s.nan_count} NaN`}</span>`
            : `<span class="stat muted">no statistics</span>`;

        return `
        <div class="var-row">
            <div class="var-name">
                <code>${v.name}${v.long_name ? " – " + v.long_name : ""}</code>
                <span class="var-meta">${v.dtype} ${v.shape.join("×")}</span>
            </div>
            <div class="var-stats">${statsHtml}</div>
        </div>`;
    }).join("");

    return `
    <div class="detail-section">
        <div class="detail-section-title">Variables</div>
        <div class="var-table">${rows}</div>
    </div>`;
}

function fmt(val) {
    if (val === null || val === undefined) return "—";
    const abs = Math.abs(val);
    if (abs === 0) return "0";
    if (abs >= 1000 || (abs < 0.01 && abs > 0)) return val.toExponential(2);
    return val.toFixed(3);
}

function renderDetail(ds) {
    const sp      = ds.spatial  || {};
    const tp      = ds.temporal || {};
    const fmtStr  = (ds.format || "").toLowerCase();   
    const name    = (ds.source || ds.dataset_id).split(/[\\/]/).pop();
    const varsHtml = renderVariables(ds.variables);    

    const globalAttrs = Object.entries(ds.global_attributes || {})
        .filter(([, v]) => v && String(v).trim())
        .slice(0, 12)
        .map(([k, v]) => `<div class="detail-k">${k}</div><div class="detail-v">${v}</div>`)
        .join("");

    const dims = Object.entries(ds.dimensions || {})
        .map(([k, v]) => `<span class="var-chip">${k}: ${v}</span>`)
        .join("");

    return `
    <div class="detail-section">
        <div class="detail-title">${name}</div>
        <div class="detail-id">${ds.dataset_id}</div>
        <span class="badge ${fmtStr}">${fmtStr}</span>
    </div>

    <div class="detail-section">
        <div class="detail-section-title">Spatial</div>
        <div class="detail-kv">
            <div class="detail-k">Lat range</div>
            <div class="detail-v accent">${sp.lat_min != null ? `${sp.lat_min.toFixed(4)} → ${sp.lat_max.toFixed(4)}` : "—"}</div>
            <div class="detail-k">Lon range</div>
            <div class="detail-v accent">${sp.lon_min != null ? `${sp.lon_min.toFixed(4)} → ${sp.lon_max.toFixed(4)}` : "—"}</div>
            ${sp.crs ? `<div class="detail-k">CRS</div><div class="detail-v">${sp.crs}</div>` : ""}
        </div>
    </div>

    <div class="detail-section">
        <div class="detail-section-title">Temporal</div>
        <div class="detail-kv">
            <div class="detail-k">Start</div>
            <div class="detail-v accent">${tp.start ? tp.start.slice(0, 19) : "—"}</div>
            <div class="detail-k">End</div>
            <div class="detail-v accent">${tp.end ? tp.end.slice(0, 19) : "—"}</div>
        </div>
    </div>

    ${varsHtml}

    ${dims ? `
    <div class="detail-section">
        <div class="detail-section-title">Dimensions</div>
        ${dims}
    </div>` : ""}

    ${globalAttrs ? `
    <div class="detail-section">
        <div class="detail-section-title">Global Attributes</div>
        <div class="detail-kv">${globalAttrs}</div>
    </div>` : ""}

    <div class="detail-delete">
        <button class="btn-danger detail-delete-btn">Delete this dataset</button>
    </div>`;
}

document.getElementById("detail-close").addEventListener("click", () => {
    document.getElementById("detail-overlay").classList.add("hidden");
});

document.getElementById("detail-overlay").addEventListener("click", e => {
    if (e.target === document.getElementById("detail-overlay")) {
        document.getElementById("detail-overlay").classList.add("hidden");
    }
});

// ── DELETE ────────────────────────────────────────────────────

async function confirmDelete(datasetId, context) {
    if (!confirm(`Delete dataset ${datasetId.slice(0, 12)}…? This cannot be undone.`)) return;

    try {
        const data = await API.delete(`/datasets/${datasetId}`);
        if (data.error) { showToast(`Error: ${data.error}`); return; }

        showToast("Dataset deleted.");

        if (context === "detail") {
            document.getElementById("detail-overlay").classList.add("hidden");
            loadBrowse();
        } else {
            // Remove card from result list
            document.querySelectorAll(`[data-id="${datasetId}"]`).forEach(el => el.remove());
        }
    } catch {
        showToast("Delete failed.");
    }
}

// ── HELPERS ───────────────────────────────────────────────────

function setLoading(btn, loading) {
    btn.disabled = loading;
    if (loading) {
        btn._originalHTML = btn.innerHTML;
        btn.innerHTML = `<span class="spinner"></span>`;
    } else {
        btn.innerHTML = btn._originalHTML || btn.innerHTML;
    }
}
