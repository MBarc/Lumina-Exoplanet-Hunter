/* Lumina candidate detail page: numbers, transit plots, and how to reproduce. */
"use strict";

const API_BASE = ["localhost", "127.0.0.1"].includes(location.hostname)
  ? "http://localhost:18000"
  : "https://lumina-exoplanet-hunter.onrender.com";

// Every value from the API is untrusted (it comes from volunteer machines):
// always escape before it touches HTML.
const esc = v => String(v ?? "").replace(/[&<>"']/g, ch =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));

const STAR = { kepler: "KIC", k2: "EPIC", tess: "TIC" };
const BADGE = {
  new: ["NEW", "badge-new"],
  unchecked: ["NOT YET CHECKED", "badge-known"],
  known_planet: ["KNOWN PLANET", "badge-known"],
  known_candidate: ["KNOWN CANDIDATE", "badge-known"],
  known_false_positive: ["KNOWN FALSE POSITIVE", "badge-fp"],
};

// Star ids are digits (enforced by the API); re-parse anyway because they end
// up in links and in code a reviewer may run.
const starNum = c => { const n = parseInt(c.tic_id, 10); return Number.isFinite(n) ? n : null; };

function starLabel(c) {
  return `${STAR[(c.mission || "").toLowerCase()] || "STAR"} ${starNum(c) ?? "?"}`;
}

function badge(c) {
  const cat = c.catalog || { status: "unchecked" };
  let [text, cls] = BADGE[cat.status] || BADGE.unchecked;
  if (cat.status === "new" && cat.known_star) text = "NEW ON KNOWN STAR";
  if (cat.name) text += cat.alias ? ` · alias of ${cat.name}` : ` · ${cat.name}`;
  return [text, cls];
}

// Minimal SVG line plot. series: [{y: number[], color}]. range: fixed [lo, hi]
// so diagnostics share the transit's scale (auto-scaling would blow pure noise
// up to look like a dip).
function plot(series, { w = 520, h = 200, mark = null, range = null } = {}) {
  const ys = series.flatMap(s => s.y).filter(Number.isFinite);
  if (!ys.length) return `<div class="empty-state">NOT SENT FOR THIS CANDIDATE</div>`;
  const [lo, hi] = range || [Math.min(...ys), Math.max(...ys)], pad = 14;
  const sx = (i, n) => pad + (i / Math.max(n - 1, 1)) * (w - 2 * pad);
  const sy = v => pad + (hi - v) / ((hi - lo) || 1) * (h - 2 * pad);
  const lines = series.map(s => {
    const pts = s.y.map((v, i) => Number.isFinite(v) ? `${sx(i, s.y.length).toFixed(1)},${sy(v).toFixed(1)}` : null)
      .filter(Boolean).join(" ");
    return `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="1.6"/>`;
  }).join("");
  const midline = mark === "center" ? `<line x1="${w / 2}" y1="${pad}" x2="${w / 2}" y2="${h - pad}" class="plot-mark"/>` : "";
  return `<svg viewBox="0 0 ${w} ${h}" class="plot" role="img">${midline}${lines}</svg>`;
}

const fmt = (v, d = 4) => (v === null || v === undefined || !Number.isFinite(v)) ? "—" : v.toFixed(d);

function row(label, value, help = "") {
  return `<tr><th>${esc(label)}</th><td>${value}</td><td class="help">${esc(help)}</td></tr>`;
}

function render(c) {
  const [text, cls] = badge(c);
  document.title = `${starLabel(c)} · Lumina Candidate`;
  document.getElementById("d-star").textContent = starLabel(c);
  const b = document.getElementById("d-badge");
  b.textContent = text;
  b.className = `badge ${cls}`;
  const found = c.reported_at ? new Date(c.reported_at).toISOString().slice(0, 16).replace("T", " ") + " UTC" : "—";
  document.getElementById("d-sub").textContent =
    `${(c.mission || "").toUpperCase()} · found ${found} by ${c.finder || c.worker_hostname || "an anonymous volunteer"}`;

  const lv = (c.local_view || []).filter(Number.isFinite);
  const pad = lv.length ? 0.15 * (Math.max(...lv) - Math.min(...lv)) : 0;
  const transitRange = lv.length ? [Math.min(...lv) - pad, Math.max(...lv) + pad] : null;
  document.getElementById("p-local").innerHTML = plot([{ y: c.local_view || [], color: "#00c8ff" }], { mark: "center" });
  document.getElementById("p-global").innerHTML = plot([{ y: c.global_view || [], color: "#00c8ff" }], { mark: "center" });
  document.getElementById("p-oddeven").innerHTML = plot([
    { y: c.odd_view || [], color: "#06d6a0" }, { y: c.even_view || [], color: "#ffd166" }],
    { mark: "center", range: transitRange });
  document.getElementById("p-secondary").innerHTML = plot([{ y: c.secondary_view || [], color: "#00c8ff" }],
    { mark: "center", range: transitRange });

  const cat = c.catalog || {};
  const hours = Number.isFinite(c.duration_days) ? c.duration_days * 24 : null;
  const sha = c.model_sha256 ? `<code>${esc(c.model_sha256.slice(0, 12))}</code>` : "—";
  const file = c.fits_url
    ? `<a href="https://mast.stsci.edu/api/v0.1/Download/file?uri=${encodeURIComponent(c.fits_url)}" rel="noopener">${esc(c.fits_url.split("/").pop())}</a>`
    : "—";
  document.getElementById("numbers-table").innerHTML = [
    row("Period", `${fmt(c.period_days, 5)} days`, "time between transits"),
    row("Mid-transit time (t0)", `${fmt(c.t0, 5)}`, "in the mission's own clock (Kepler BKJD / TESS BTJD)"),
    row("Mid-transit time (BJD)", `${fmt(c.t0_bjd, 5)}`, "same instant, Barycentric Julian Date (TDB)"),
    row("Duration", `${fmt(hours, 2)} hours`, "how long each dip lasts"),
    row("Depth", Number.isFinite(c.depth_ppm) ? `${Math.round(c.depth_ppm).toLocaleString()} ppm` : "—",
        "parts per million of the star's light blocked"),
    row("Signal-to-noise", fmt(c.snr, 1), "depth vs. noise over all in-transit points"),
    row("Transits observed", fmt(c.n_transits, 0), ""),
    row("Odd/even difference", Number.isFinite(c.odd_even_diff_ppm) ? `${Math.round(c.odd_even_diff_ppm).toLocaleString()} ppm` : "—",
        "depth difference between odd and even transits; large values suggest an eclipsing binary"),
    row("Secondary eclipse depth", Number.isFinite(c.secondary_depth_ppm) ? `${Math.round(c.secondary_depth_ppm).toLocaleString()} ppm` : "—",
        "a dip half an orbit later suggests two stars"),
    row("Centroid shift", `${fmt(c.centroid_shift, 4)} px`, "the star's apparent position moving during the dip suggests a neighbour"),
    row("ExoNet score", `${fmt((c.exonet_score || 0) * 100, 1)}%`, "the model's planet-likeness score"),
    row("Catalogue match", cat.status ? esc(badge(c)[0]) : "not checked yet",
        cat.catalog_period ? `catalogued period ${fmt(cat.catalog_period, 5)} d` : "checked against NASA's KOI, TOI and K2 lists"),
    row("Status", c.verified ? "Verified" : "Unverified", "verified = reproduced centrally and checked by a person"),
    row("Data file", file, "the NASA light curve this was found in"),
    row("Model", sha, "fingerprint of the model that scored it"),
  ].join("");

  const mission = (c.mission || "").toLowerCase();
  const sid = starNum(c);
  const target = `${STAR[mission] || "KIC"} ${sid ?? 0}`;
  // Same product the node used: Kepler/K2 long cadence; TESS 2-min SPOC when the
  // file was a 2-min light curve, otherwise let lightkurve pick.
  const search = mission === "tess"
    ? (/-s_lc\.fits$/.test(c.fits_url || "") ? `author="SPOC", exptime=120` : "")
    : `author="${mission === "k2" ? "K2" : "Kepler"}", exptime=1800`;
  const P = fmt(c.period_days, 6), D = fmt(c.duration_days, 5);
  const hasT0 = Number.isFinite(c.t0);
  document.getElementById("d-code").textContent = [
    "import lightkurve as lk",
    `lc = lk.search_lightcurve("${target}"${search ? ", " + search : ""}).download_all().stitch()`,
    ...(hasT0 ? [
      `T0 = ${fmt(c.t0, 6)}  # mission clock, same as lc.time`,
      `mask = lc.create_transit_mask(period=${P}, transit_time=T0, duration=${D})`,
      `lc.flatten(mask=mask).fold(period=${P}, epoch_time=T0).scatter()`,
    ] : [`lc.flatten().fold(period=${P}).scatter()  # no transit time recorded`]),
  ].join("\n");

  const id = encodeURIComponent(sid ?? "");
  const exofop = mission === "tess"
    ? `https://exofop.ipac.caltech.edu/tess/target.php?id=${id}`
    : mission === "kepler" ? `https://exofop.ipac.caltech.edu/kepler/edit_target.php?id=${id}` : null;
  document.getElementById("d-links").innerHTML = [
    `<a href="https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery=${encodeURIComponent(target)}" rel="noopener">MAST</a>`,
    exofop ? `<a href="${exofop}" rel="noopener">ExoFOP</a>` : null,
    `<a href="https://simbad.u-strasbg.fr/simbad/sim-id?Ident=${encodeURIComponent(target)}" rel="noopener">SIMBAD</a>`,
  ].filter(Boolean).join(" · ");

  for (const s of ["detail-head", "plots", "numbers", "reproduce"]) document.getElementById(s).hidden = false;
}

(async function main() {
  const id = new URLSearchParams(location.search).get("id");
  const err = document.getElementById("detail-error");
  if (!id || !/^[0-9a-f]{24}$/.test(id)) {
    err.textContent = "NO CANDIDATE SELECTED";
    err.hidden = false;
    return;
  }
  try {
    const res = await fetch(`${API_BASE}/candidates/${id}`, { cache: "no-store" });
    if (!res.ok) throw new Error(res.status === 404 ? "CANDIDATE NOT FOUND" : `API ERROR ${res.status}`);
    render(await res.json());
  } catch (e) {
    err.textContent = e.message || "COULD NOT REACH MISSION CONTROL";
    err.hidden = false;
  }
})();
