from __future__ import annotations
import numpy as np
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from dash import Input, Output, html
from dash.exceptions import PreventUpdate
from dashboard.app import app
from dashboard import db

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
    Output("live-tic-label", "children"),
    Output("live-curve-graph", "figure"),
    Output("curve-stats-period", "children"),
    Output("curve-stats-depth", "children"),
    Output("curve-stats-bls", "children"),
    Input("interval-5s", "n_intervals"),
)
def update_live_curve(n):
    candidate = db.get_top_candidate_with_views()
    if not candidate:
        empty_fig = go.Figure(layout=_DARK_LAYOUT)
        return "AWAITING DATA", empty_fig, "\u2014", "\u2014", "\u2014"

    global_view = candidate.get("global_view") or []
    if not global_view:
        empty_fig = go.Figure(layout=_DARK_LAYOUT)
        return "AWAITING DATA", empty_fig, "\u2014", "\u2014", "\u2014"

    x = np.linspace(-0.5, 0.5, len(global_view)).tolist()
    y = global_view
    fig = go.Figure(
        data=[go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="#00c8ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,200,255,0.04)",
        )],
        layout=_DARK_LAYOUT,
    )
    tic_label = f"{candidate.get('tic_id', 'UNKNOWN')}  \u00b7  SECTOR {candidate.get('sector', '?')}"
    period = f"{candidate['period_days']:.3f} d" if candidate.get("period_days") is not None else "\u2014"
    depth  = f"{candidate['depth_ppm']:.0f} ppm" if candidate.get("depth_ppm") is not None else "\u2014"
    bls    = f"{candidate['bls_power']:.1f}" if candidate.get("bls_power") is not None else "\u2014"
    return tic_label, fig, period, depth, bls


@app.callback(
    Output("network-nodes", "children"),
    Output("network-stars", "children"),
    Output("network-candidates", "children"),
    Output("network-queued", "children"),
    Output("node-count-badge", "children"),
    Input("interval-5s", "n_intervals"),
)
def update_network_stats(n):
    stats = db.get_network_stats()
    nodes      = stats.get("active_nodes", 0)
    stars      = stats.get("total_stars_analyzed", 0)
    candidates = stats.get("total_candidates", 0)
    queued     = stats.get("sectors_queued", 0)
    badge = f"{nodes} NODE{'S' if nodes != 1 else ''}"
    return (
        f"{nodes:,}", f"{stars:,}", f"{candidates:,}", f"{queued:,}",
        badge,
    )


@app.callback(
    Output("detail-title-badge", "children"),
    Output("detail-title-badge", "className"),
    Output("detail-curve-graph", "figure"),
    Input("interval-5s", "n_intervals"),
)
def update_detail(n):
    candidate = db.get_top_candidate_with_views()
    if not candidate:
        return "NO DATA", "score-badge badge-false", go.Figure(layout=_DARK_LAYOUT)

    local_view = candidate.get("local_view") or []
    if not local_view:
        return "NO DATA", "score-badge badge-false", go.Figure(layout=_DARK_LAYOUT)

    score = candidate.get("exonet_score", 0) or 0
    n_local = len(local_view)
    x = np.linspace(-0.4, 0.4, n_local).tolist()

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
    if score >= 0.8:
        badge_text  = f"{score * 100:.0f}%  STRONG CANDIDATE"
        badge_class = "score-badge badge-strong"
    elif score >= 0.5:
        badge_text  = f"{score * 100:.0f}%  CANDIDATE"
        badge_class = "score-badge badge-candidate"
    else:
        badge_text  = f"{score * 100:.0f}%  FALSE POSITIVE"
        badge_class = "score-badge badge-false"
    return badge_text, badge_class, fig


@app.callback(
    Output("node-list", "children"),
    Input("interval-10s", "n_intervals"),
)
def update_node_list(n):
    nodes = db.get_node_telemetry()
    if not nodes:
        return html.Div("NO ACTIVE NODES", className="empty-state")

    now = datetime.now(timezone.utc)
    rows = []
    for node in nodes:
        reported = node.get("reported_at")
        if reported and reported.tzinfo is None:
            reported = reported.replace(tzinfo=timezone.utc)
        age_s = (now - reported).total_seconds() if reported else 999
        if age_s < 60:
            dot_class = "status-dot dot-green"
        elif age_s < 300:
            dot_class = "status-dot dot-amber"
        else:
            dot_class = "status-dot dot-red"

        uptime_s = int(node.get("uptime_seconds", 0) or 0)
        d, rem = divmod(uptime_s, 86400)
        h, rem = divmod(rem, 3600)
        m      = rem // 60
        uptime_str = f"{d}d {h}h {m}m" if d else f"{h}h {m}m"

        cpu = min(int(node.get("cpu_percent", 0) or 0), 100)
        tic = node.get("current_tic_id") or "\u2014"

        rows.append(html.Div(className="node-row", children=[
            html.Div(className=dot_class),
            html.Div(children=[
                html.Div(node.get("hostname", "unknown"), className="node-hostname"),
                html.Div(
                    f"{uptime_str}  \u00b7  {node.get('stars_analyzed', 0)} stars  \u00b7  TIC {tic}",
                    className="node-meta",
                ),
                html.Div(className="cpu-bar-bg", children=[
                    html.Div(className="cpu-bar-fg", style={"width": f"{cpu}%"}),
                ]),
            ]),
            html.Div(
                f"{cpu}%",
                style={"fontFamily": "Consolas,monospace", "fontSize": "10px", "color": "#4a6380"},
            ),
        ]))
    return rows


@app.callback(
    Output("candidates-table", "data"),
    Input("interval-5s", "n_intervals"),
)
def update_candidates_table(n):
    rows = db.get_recent_candidates(20)
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
