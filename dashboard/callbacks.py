from __future__ import annotations
import socket
import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from dashboard.app import app
from dashboard import db

# Resolve this machine's hostname once at startup — all DB queries are scoped to it.
_HOSTNAME = socket.gethostname()

_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=4, r=4, t=4, b=4),
    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False),
)


@app.callback(Output("header-clock", "children"), Input("interval-1s", "n_intervals"))
def update_clock(n):
    return datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S  UTC")


@app.callback(
    Output("live-tic-label",      "children"),
    Output("live-curve-graph",    "figure"),
    Output("curve-stats-period",  "children"),
    Output("curve-stats-depth",   "children"),
    Output("curve-stats-bls",     "children"),
    Input("interval-5s", "n_intervals"),
)
def update_currently_processing(n):
    """Show the most recent candidate this machine submitted as the active target."""
    candidate = db.get_my_latest_candidate(_HOSTNAME)
    empty_fig  = go.Figure(layout=_DARK_LAYOUT)

    if not candidate:
        return "AWAITING SIGNAL", empty_fig, "—", "—", "—"

    global_view = candidate.get("global_view") or []
    if not global_view:
        return "AWAITING SIGNAL", empty_fig, "—", "—", "—"

    x = np.linspace(-0.5, 0.5, len(global_view)).tolist()
    fig = go.Figure(
        data=[go.Scatter(
            x=x, y=global_view, mode="lines",
            line=dict(color="#00c8ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,200,255,0.04)",
        )],
        layout=_DARK_LAYOUT,
    )
    tic_label = f"{candidate.get('tic_id', 'UNKNOWN')}  ·  SECTOR {candidate.get('sector', '?')}"
    period = f"{candidate['period_days']:.3f} d"   if candidate.get("period_days")  is not None else "—"
    depth  = f"{candidate['depth_ppm']:.0f} ppm"   if candidate.get("depth_ppm")    is not None else "—"
    bls    = f"{candidate['bls_power']:.1f}"        if candidate.get("bls_power")    is not None else "—"
    return tic_label, fig, period, depth, bls


@app.callback(
    Output("contrib-stars",      "children"),
    Output("contrib-candidates", "children"),
    Output("contrib-uptime",     "children"),
    Output("contrib-best",       "children"),
    Output("status-badge",       "children"),
    Output("status-badge",       "className"),
    Input("interval-5s", "n_intervals"),
)
def update_my_contributions(n):
    """Pull personal stats for this machine and update the contribution counters."""
    stats = db.get_my_stats(_HOSTNAME)

    stars      = stats.get("stars_analyzed",  0)
    candidates = stats.get("candidates_found", 0)
    uptime_s   = int(stats.get("uptime_seconds", 0))
    best_score = stats.get("best_score", 0.0)

    # Format uptime as "Xh Ym" or "Xd Yh"
    d, rem  = divmod(uptime_s, 86400)
    h, rem  = divmod(rem, 3600)
    m       = rem // 60
    if d:
        uptime_str = f"{d}d {h}h"
    elif h:
        uptime_str = f"{h}h {m}m"
    else:
        uptime_str = f"{m}m"

    best_str = f"{best_score * 100:.0f}%" if best_score else "—"

    # Badge reflects whether we have recent telemetry (i.e. the worker is running)
    telemetry = db.get_my_telemetry(_HOSTNAME)
    if telemetry:
        reported = telemetry.get("reported_at")
        if reported and reported.tzinfo is None:
            reported = reported.replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - reported).total_seconds() if reported else 999
        if age_s < 60:
            badge_text, badge_class = "ONLINE", "node-badge badge-online"
        elif age_s < 300:
            badge_text, badge_class = "IDLE",   "node-badge badge-idle"
        else:
            badge_text, badge_class = "OFFLINE", "node-badge badge-offline"
    else:
        badge_text, badge_class = "NO DATA", "node-badge"

    return f"{stars:,}", f"{candidates:,}", uptime_str, best_str, badge_text, badge_class


@app.callback(
    Output("my-machine-detail", "children"),
    Input("interval-10s", "n_intervals"),
)
def update_my_machine(n):
    """Render detailed telemetry for this machine in the right-hand panel."""
    telemetry = db.get_my_telemetry(_HOSTNAME)
    if not telemetry:
        return html.Div("CONNECTING...", className="empty-state")

    now = datetime.now(timezone.utc)
    reported = telemetry.get("reported_at")
    if reported and reported.tzinfo is None:
        reported = reported.replace(tzinfo=timezone.utc)
    age_s = (now - reported).total_seconds() if reported else 999

    if age_s < 60:
        dot_class, status_text = "status-dot dot-green", "ONLINE"
    elif age_s < 300:
        dot_class, status_text = "status-dot dot-amber", "IDLE"
    else:
        dot_class, status_text = "status-dot dot-red",   "OFFLINE"

    cpu  = min(int(telemetry.get("cpu_percent", 0) or 0), 100)
    ram  = min(int(telemetry.get("ram_percent", 0) or 0), 100)
    tic  = telemetry.get("current_tic_id") or "—"
    sec  = telemetry.get("current_sector")

    uptime_s = int(telemetry.get("uptime_seconds", 0) or 0)
    d, rem = divmod(uptime_s, 86400)
    h, rem = divmod(rem, 3600)
    m      = rem // 60
    uptime_str = f"{d}d {h}h {m}m" if d else f"{h}h {m}m"

    current_target = f"TIC {tic}"
    if sec is not None:
        current_target += f"  ·  SECTOR {sec}"

    def _bar(pct, color):
        return html.Div(className="machine-bar-bg", children=[
            html.Div(className="machine-bar-fg", style={"width": f"{pct}%", "background": color}),
        ])

    def _metric(label, value, bar=None):
        children = [
            html.Div(className="machine-metric-row", children=[
                html.Span(label, className="machine-metric-label"),
                html.Span(value, className="machine-metric-value"),
            ]),
        ]
        if bar is not None:
            children.append(bar)
        return html.Div(className="machine-metric", children=children)

    return html.Div(className="machine-detail", children=[
        html.Div(className="machine-status-row", children=[
            html.Div(className=dot_class),
            html.Span(_HOSTNAME.upper(), className="node-hostname"),
            html.Span(status_text, className="machine-status-text"),
        ]),
        html.Div(className="machine-divider"),
        _metric("UPTIME",   uptime_str),
        _metric("CURRENT",  current_target),
        html.Div(className="machine-divider"),
        _metric("CPU",  f"{cpu}%",  _bar(cpu, "#00c8ff")),
        _metric("RAM",  f"{ram}%",  _bar(ram, "#a855f7")),
    ])


@app.callback(
    Output("detail-title-badge", "children"),
    Output("detail-title-badge", "className"),
    Output("detail-curve-graph", "figure"),
    Input("interval-5s", "n_intervals"),
)
def update_best_find(n):
    """Show the highest-scoring transit candidate this machine has found."""
    candidate = db.get_my_best_candidate(_HOSTNAME)
    empty_fig  = go.Figure(layout=_DARK_LAYOUT)

    if not candidate:
        return "NO DATA", "score-badge badge-false", empty_fig

    local_view = candidate.get("local_view") or []
    if not local_view:
        return "NO DATA", "score-badge badge-false", empty_fig

    score   = candidate.get("exonet_score", 0) or 0
    tic_id  = candidate.get("tic_id", "UNKNOWN")
    sector  = candidate.get("sector", "?")
    x = np.linspace(-0.4, 0.4, len(local_view)).tolist()

    fig = go.Figure(
        data=[go.Scatter(
            x=x, y=local_view, mode="lines",
            line=dict(color="#06d6a0", width=1.5),
            fill="tozeroy", fillcolor="rgba(6,214,160,0.06)",
        )],
        layout={
            **_DARK_LAYOUT,
            "shapes": [{
                "type": "line", "x0": 0, "x1": 0,
                "y0": 0, "y1": 1, "yref": "paper",
                "line": {"color": "#ffd166", "width": 1, "dash": "dot"},
            }],
        },
    )

    pct = score * 100
    if score >= 0.8:
        badge_text  = f"{pct:.0f}%  STRONG CANDIDATE  ·  {tic_id}"
        badge_class = "score-badge badge-strong"
    elif score >= 0.5:
        badge_text  = f"{pct:.0f}%  CANDIDATE  ·  {tic_id}"
        badge_class = "score-badge badge-candidate"
    else:
        badge_text  = f"{pct:.0f}%  FALSE POSITIVE  ·  {tic_id}"
        badge_class = "score-badge badge-false"

    return badge_text, badge_class, fig


@app.callback(
    Output("my-queue-content", "children"),
    Input("interval-10s", "n_intervals"),
)
def update_my_queue(n):
    """Show global queue depth as labelled bars in the MY QUEUE panel."""
    qs = db.get_my_queue_status()
    queued   = qs.get("queued",   0)
    assigned = qs.get("assigned", 0)
    done     = qs.get("done",     0)
    total    = max(queued + assigned + done, 1)  # avoid div-by-zero

    def _bar_item(label, count, color, denominator=total):
        pct = min(int(count / denominator * 100), 100)
        return html.Div(className="queue-stat-item", children=[
            html.Div(className="queue-stat-header", children=[
                html.Span(label,       className="queue-stat-label"),
                html.Span(f"{count:,}", className="queue-stat-value"),
            ]),
            html.Div(className="queue-bar-bg", children=[
                html.Div(className="queue-bar-fg",
                         style={"width": f"{pct}%", "background": color}),
            ]),
        ])

    return html.Div(className="queue-stat-row", children=[
        _bar_item("QUEUED",   queued,   "#00c8ff"),
        _bar_item("ASSIGNED", assigned, "#ffd166"),
        _bar_item("DONE",     done,     "#06d6a0"),
    ])


@app.callback(
    Output("my-history-content", "children"),
    Input("interval-10s", "n_intervals"),
)
def update_my_history(n):
    """Populate the MY HISTORY panel with recent processed stars from this node."""
    rows = db.get_my_history(_HOSTNAME, 10)
    if not rows:
        return html.Div("NO HISTORY YET", className="empty-state")

    items = []
    for row in rows:
        tic    = row.get("tic_id", "?")
        mission = (row.get("mission") or "").upper()
        sector = row.get("sector")
        cands  = row.get("candidates_found", 0)
        ts     = row.get("processed_at")

        label = f"TIC {tic}"
        if sector is not None:
            label += f" · S{sector}"
        elif mission:
            label += f" · {mission}"

        time_str = ts.strftime("%H:%M") if ts and hasattr(ts, "strftime") else "—"
        cand_str = f"✓ {cands}" if cands else ""

        items.append(html.Div(className="history-item", children=[
            html.Span(label,    className="history-item-tic"),
            html.Span(f"{time_str}  {cand_str}".strip(), className="history-item-time"),
        ]))

    return html.Div(className="history-list", children=items)


@app.callback(
    Output("candidates-table", "data"),
    Input("interval-5s", "n_intervals"),
)
def update_my_findings(n):
    """Populate the findings table with candidates from this machine only."""
    rows = db.get_my_candidates(_HOSTNAME, 20)
    for r in rows:
        if "exonet_score" in r and r["exonet_score"] is not None:
            r["exonet_score"] = f"{r['exonet_score'] * 100:.1f}"
        if "period_days" in r and r["period_days"] is not None:
            r["period_days"] = f"{r['period_days']:.3f} d"
        if "duration_days" in r and r["duration_days"] is not None:
            r["duration_days"] = f"{r['duration_days']:.2f} d"
        if "depth_ppm" in r and r["depth_ppm"] is not None:
            r["depth_ppm"] = f"{r['depth_ppm']:.0f} ppm"
        if "reported_at" in r and hasattr(r["reported_at"], "strftime"):
            r["reported_at"] = r["reported_at"].strftime("%Y-%m-%d %H:%M")
        if "_id" in r:
            r["_id"] = str(r["_id"])
    return rows
