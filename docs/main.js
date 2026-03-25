/**
 * Lumina Mission Control — GitHub Pages frontend
 *
 * Fetches live network stats from the Lumina REST API and renders them
 * into the page. When the API is unreachable (or not yet deployed) the
 * page degrades gracefully, showing placeholder dashes and an OFFLINE badge.
 *
 * API contract
 * ------------
 * All endpoints return JSON. The base URL is configured in API_BASE below.
 *
 * GET /api/stats
 *   Returns global network counters.
 *   {
 *     active_nodes:    number,   // nodes with a heartbeat in the last 5 min
 *     stars_analyzed:  number,   // total across all nodes, all time
 *     candidates_found: number,  // total candidates reported
 *     compute_hours:   number    // total CPU-hours contributed
 *   }
 *
 * GET /api/candidates?limit=10
 *   Returns the most recent high-scoring candidates.
 *   [
 *     {
 *       tic_id:       string,
 *       sector:       number,
 *       period_days:  number,
 *       depth_ppm:    number,
 *       exonet_score: number,   // 0.0 – 1.0
 *       worker_node:  string,   // hostname (may be anonymised)
 *       reported_at:  string    // ISO 8601
 *     },
 *     ...
 *   ]
 *
 * GET /api/leaderboard?limit=10
 *   Returns top contributors ranked by stars_analyzed.
 *   [
 *     {
 *       rank:           number,
 *       hostname:       string,  // may be anonymised / display name
 *       stars_analyzed: number,
 *       candidates_found: number
 *     },
 *     ...
 *   ]
 *
 * GET /api/activity?hours=24
 *   Returns candidate counts per hour for the last N hours (for the sparkline).
 *   [
 *     { hour: string, count: number },  // hour is ISO 8601
 *     ...
 *   ]
 */

"use strict";

// ── Configuration ──────────────────────────────────────────────────────────

/**
 * Base URL for the Lumina REST API.
 * Update this when the API is deployed.
 */
const API_BASE = "https://lumina-exoplanet-hunter.onrender.com";

/** How often to poll the API for fresh data (milliseconds). */
const POLL_INTERVAL_MS = 30_000;

// ── Clock ──────────────────────────────────────────────────────────────────

function updateClock() {
  const el = document.getElementById("clock");
  if (el) el.textContent = new Date().toISOString().replace("T", "  ").slice(0, 22) + "  UTC";
}
setInterval(updateClock, 1000);
updateClock();

// ── Starfield canvas ───────────────────────────────────────────────────────

(function initStarfield() {
  const canvas = document.getElementById("starfield");
  const ctx    = canvas.getContext("2d");
  let stars    = [];

  function resize() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function generateStars(n) {
    stars = Array.from({ length: n }, () => ({
      x:    Math.random() * canvas.width,
      y:    Math.random() * canvas.height,
      r:    Math.random() * 1.2 + 0.2,
      // Slight cyan tint on a fraction of stars to match the theme
      hue:  Math.random() < 0.15 ? "rgba(0,200,255," : "rgba(255,255,255,",
      a:    Math.random() * 0.6 + 0.15,
      drift: (Math.random() - 0.5) * 0.015,
    }));
  }

  function drawStars() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    stars.forEach(s => {
      s.y += s.drift;
      // Wrap stars that drift off the top/bottom
      if (s.y < 0) s.y = canvas.height;
      if (s.y > canvas.height) s.y = 0;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = s.hue + s.a + ")";
      ctx.fill();
    });
    requestAnimationFrame(drawStars);
  }

  window.addEventListener("resize", () => { resize(); generateStars(180); });
  resize();
  generateStars(180);
  drawStars();
})();

// ── Ticker ─────────────────────────────────────────────────────────────────

function buildTicker(stats) {
  const items = [
    ["NODES",      stats ? formatNumber(stats.active_nodes) + " ACTIVE"    : "—"],
    ["STARS",      stats ? formatNumber(stats.stars_analyzed) + " ANALYZED" : "—"],
    ["CANDIDATES", stats ? formatNumber(stats.candidates_found) + " FOUND"  : "—"],
    ["COMPUTE",    stats ? formatHours(stats.compute_hours) + " CONTRIBUTED": "—"],
    ["MODEL",      stats && stats.model_version ? stats.model_version.toUpperCase() : "EXONET v2.0"],
    ["QUEUE",      stats ? formatNumber(stats.queue_remaining) + " TARGETS REMAINING" : "—"],
    ["ALGORITHM",  "BLS + RESIDUAL CNN"],
    ["MISSIONS",   "KEPLER  ·  K2  ·  TESS"],
    ["TARGET",     "EXOPLANET TRANSIT DETECTION"],
  ];

  // Duplicate for seamless infinite scroll
  const html = [...items, ...items].map(([label, value]) =>
    `<span class="ticker-item">` +
    `<span style="color:#2a4060">${label}: </span>` +
    `<span class="ticker-highlight">${value}</span>` +
    `</span><span class="ticker-sep">◆</span>`
  ).join("");

  document.getElementById("ticker-content").innerHTML = html;
}

// ── Number formatting helpers ───────────────────────────────────────────────

function formatNumber(n) {
  if (n === undefined || n === null) return "—";
  return Number(n).toLocaleString();
}

function formatHours(h) {
  if (!h) return "—";
  if (h >= 8760) return (h / 8760).toFixed(1) + "k";
  if (h >= 1000) return (h / 1000).toFixed(1) + "k";
  return Math.round(h).toString();
}

// Counter animation — smoothly ticks a stat number up to its target value
function animateStat(el, target) {
  const start    = parseInt(el.textContent.replace(/[^0-9]/g, "")) || 0;
  const duration = 800;
  const startTs  = performance.now();
  function step(ts) {
    const progress = Math.min((ts - startTs) / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3);   // ease-out cubic
    const current  = Math.round(start + (target - start) * eased);
    el.textContent = formatNumber(current);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

// ── Status badge ────────────────────────────────────────────────────────────

function setStatus(state) {
  const el = document.getElementById("network-status");
  el.className = "status-badge";
  if (state === "online") {
    el.textContent = "ONLINE";
    el.classList.add("badge-online");
  } else if (state === "error") {
    el.textContent = "API OFFLINE";
    el.classList.add("badge-error");
  } else {
    el.textContent = "CONNECTING";
    el.classList.add("badge-connecting");
  }
}

// ── Render functions ────────────────────────────────────────────────────────

function renderStats(data) {
  animateStat(document.getElementById("stat-nodes"),      data.active_nodes);
  animateStat(document.getElementById("stat-stars"),      data.stars_analyzed);
  animateStat(document.getElementById("stat-candidates"), data.candidates_found);
  animateStat(document.getElementById("stat-queue"),      data.queue_remaining ?? data.queue_depth ?? 0);

  // Compute hours shown as-is with formatting
  const computeEl = document.getElementById("stat-compute");
  computeEl.textContent = formatHours(data.compute_hours);

  // Model version badge
  if (data.model_version) {
    const badge = document.getElementById("model-badge");
    if (badge) badge.textContent = data.model_version.toUpperCase();
  }
}

function renderCandidates(list) {
  const el = document.getElementById("candidate-list");
  if (!list || list.length === 0) {
    el.innerHTML = `<div class="empty-state">NO CANDIDATES YET</div>`;
    return;
  }

  el.innerHTML = list.map(c => {
    const score    = c.exonet_score || 0;
    const pct      = (score * 100).toFixed(1);
    const strong   = score >= 0.8 ? "strong" : "";
    const scoreClass = score >= 0.8 ? "score-high" : score >= 0.5 ? "score-mid" : "score-low";
    const period   = c.period_days  ? `${c.period_days.toFixed(3)}d`  : "—";
    const depth    = c.depth_ppm    ? `${Math.round(c.depth_ppm)}ppm` : "—";
    const reported = c.reported_at  ? new Date(c.reported_at).toISOString().slice(0, 16).replace("T", " ") : "—";

    return `
      <div class="candidate-row ${strong}">
        <div class="candidate-tic">TIC ${c.tic_id}</div>
        <div class="candidate-meta">
          SECTOR ${c.sector || "?"}  ·  P=${period}  ·  D=${depth}
        </div>
        <div class="candidate-score ${scoreClass}">${pct}%</div>
        <div class="candidate-node">${c.worker_node || "—"}<br>${reported}</div>
      </div>`;
  }).join("");
}

function renderLeaderboard(list) {
  const el = document.getElementById("leaderboard");
  if (!list || list.length === 0) {
    el.innerHTML = `<div class="empty-state">NO DATA YET</div>`;
    return;
  }

  // Find max stars for proportional bar widths
  const maxStars = Math.max(...list.map(r => r.stars_analyzed || 0), 1);

  el.innerHTML = list.map(r => {
    const pct      = Math.round((r.stars_analyzed / maxStars) * 100);
    const topClass = r.rank <= 3 ? "top" : "";
    return `
      <div class="lb-row">
        <div class="lb-rank ${topClass}">#${r.rank}</div>
        <div class="lb-name">${r.hostname}</div>
        <div class="lb-stars">${formatNumber(r.stars_analyzed)} ★</div>
        <div class="lb-bar-wrap">
          <div class="lb-bar-bg">
            <div class="lb-bar-fg" style="width:${pct}%"></div>
          </div>
        </div>
      </div>`;
  }).join("");
}

function renderActivityChart(data) {
  const canvas = document.getElementById("activity-chart");
  if (!canvas || !data || data.length === 0) return;

  const ctx    = canvas.getContext("2d");
  const counts = data.map(d => d.count);
  const max    = Math.max(...counts, 1);
  const W      = canvas.offsetWidth;
  const H      = canvas.offsetHeight;
  canvas.width  = W;
  canvas.height = H;

  const pad    = { top: 8, right: 8, bottom: 8, left: 8 };
  const iW     = W - pad.left - pad.right;
  const iH     = H - pad.top  - pad.bottom;
  const step   = iW / (counts.length - 1 || 1);

  // Build path
  const points = counts.map((c, i) => ({
    x: pad.left + i * step,
    y: pad.top  + iH * (1 - c / max),
  }));

  // Gradient fill under line
  const gradient = ctx.createLinearGradient(0, pad.top, 0, H - pad.bottom);
  gradient.addColorStop(0,   "rgba(0,200,255,0.25)");
  gradient.addColorStop(1,   "rgba(0,200,255,0.0)");

  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(points[points.length - 1].x, H - pad.bottom);
  ctx.lineTo(points[0].x, H - pad.bottom);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Line
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach(p => ctx.lineTo(p.x, p.y));
  ctx.strokeStyle = "#00c8ff";
  ctx.lineWidth   = 1.5;
  ctx.stroke();
}

// ── API polling ─────────────────────────────────────────────────────────────

async function fetchJSON(path) {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function refresh() {
  try {
    // Fire all four requests in parallel
    const [stats, candidates, leaderboard, activity] = await Promise.all([
      fetchJSON("/stats"),
      fetchJSON("/candidates?limit=10"),
      fetchJSON("/stats/leaderboard?limit=10"),
      fetchJSON("/stats/activity?hours=24"),
    ]);

    renderStats(stats);
    renderCandidates(candidates);
    renderLeaderboard(leaderboard);
    renderActivityChart(activity);
    buildTicker(stats);
    setStatus("online");
  } catch (err) {
    console.warn("Lumina API unreachable:", err.message);
    setStatus("error");
    // Leave existing data in place rather than wiping it on a transient error
    buildTicker(null);
  }
}

// ── Boot ────────────────────────────────────────────────────────────────────

buildTicker(null);   // populate ticker immediately with static text
refresh();           // first fetch
setInterval(refresh, POLL_INTERVAL_MS);
