import os as _os
import socket as _socket
from dash import html, dcc, dash_table

# Read CSS at import time and inject inline (Dash 4 doesn't auto-serve assets/)
_CSS_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "assets", "lumina.css")
with open(_CSS_PATH) as _f:
    _CSS = _f.read()

# Machine identity — shown in the header and used to scope all DB queries
HOSTNAME = _socket.gethostname().upper()

CANDIDATE_COLUMNS = [
    {"name": "TIC ID",    "id": "tic_id"},
    {"name": "Sector",    "id": "sector"},
    {"name": "Period",    "id": "period_days"},
    {"name": "Duration",  "id": "duration_days"},
    {"name": "Depth",     "id": "depth_ppm"},
    {"name": "BLS Power", "id": "bls_power"},
    {"name": "Score",     "id": "exonet_score"},
    {"name": "Label",     "id": "label"},
    {"name": "Reported",  "id": "reported_at"},
]

TABLE_STYLE_HEADER = {
    "backgroundColor": "#0a1a30",
    "color": "#00c8ff",
    "fontFamily": "Consolas, monospace",
    "fontSize": "10px",
    "letterSpacing": "0.15em",
    "textTransform": "uppercase",
    "border": "none",
    "borderBottom": "1px solid rgba(0,200,255,0.2)",
}
TABLE_STYLE_CELL = {
    "backgroundColor": "#04090f",
    "color": "#e8f4ff",
    "fontFamily": "Consolas, monospace",
    "fontSize": "11px",
    "border": "none",
    "borderBottom": "1px solid rgba(255,255,255,0.03)",
    "padding": "6px 12px",
}
TABLE_STYLE_DATA_CONDITIONAL = [
    {
        "if": {"filter_query": "{exonet_score} >= 80"},
        "borderLeft": "2px solid #06d6a0",
        "backgroundColor": "rgba(6,214,160,0.04)",
    },
    {
        "if": {"filter_query": "{label} = FALSE_POSITIVE"},
        "color": "#4a6380",
    },
    {
        "if": {"state": "selected"},
        "backgroundColor": "rgba(0,200,255,0.08)",
        "border": "1px solid rgba(0,200,255,0.3)",
    },
]


def _make_plotly_dark_layout(**kwargs):
    """Shared dark figure layout for all graphs."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=4, r=4, t=4, b=4),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,200,255,0.06)",
            gridwidth=1,
            zeroline=False,
            showticklabels=False,
            color="#4a6380",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,200,255,0.06)",
            gridwidth=1,
            zeroline=False,
            showticklabels=False,
            color="#4a6380",
        ),
    )
    base.update(kwargs)
    return base


# Ticker items — personalized to this machine; duplicated for seamless scroll loop
_TICKER_ITEMS = [
    ("MACHINE",    HOSTNAME),
    ("STATUS",     "ONLINE"),
    ("MISSION",    "LUMINA v1.0"),
    ("MODEL",      "EXONET v2.0  ·  3M PARAMS"),
    ("ALGORITHM",  "BLS + RESIDUAL CNN"),
    ("TELESCOPE",  "KEPLER / TESS / K2"),
    ("TARGET",     "EXOPLANET TRANSIT DETECTION"),
]

def _ticker_item(label, value):
    return html.Span(className="ticker-item", children=[
        html.Span(f"{label}: ", style={"color": "#2a4060"}),
        html.Span(value, className="ticker-highlight"),
    ])

def _make_ticker():
    interleaved = []
    for label, value in _TICKER_ITEMS:
        interleaved.append(_ticker_item(label, value))
        interleaved.append(html.Span("◆", className="ticker-sep"))
    return html.Div(className="ticker-content", children=interleaved * 2)


layout = html.Div(className="app-container", children=[
    dcc.Store(id="_css-injector", data=0),

    # Intervals
    dcc.Interval(id="interval-1s",  interval=1_000,  n_intervals=0),
    dcc.Interval(id="interval-5s",  interval=5_000,  n_intervals=0),
    dcc.Interval(id="interval-10s", interval=10_000, n_intervals=0),

    # Header
    html.Div(className="header", children=[
        html.Div(className="header-left", children=[
            html.Span("◈", className="header-diamond"),
            html.Span("LUMINA", className="header-title"),
            html.Span(HOSTNAME, className="header-machine"),
        ]),
        html.Div(className="header-right", children=[
            html.Span(id="header-clock", className="utc-clock"),
            html.Span(id="status-badge", className="node-badge", children="CONNECTING"),
        ]),
    ]),

    # Main grid
    html.Div(className="main-grid", children=[

        # Panel: Currently Processing (left, spans 2 rows)
        html.Div(className="panel panel-live", children=[
            html.Div(className="panel-scanline"),
            html.Div(className="panel-title", children=[
                html.Span(className="live-dot"),
                html.Span("CURRENTLY PROCESSING", className="panel-title-text"),
            ]),
            html.Div(id="live-tic-label", className="tic-label awaiting-signal", children="AWAITING SIGNAL"),
            dcc.Graph(
                id="live-curve-graph",
                config={"displayModeBar": False},
                style={"flex": "1", "minHeight": "0"},
                figure={"data": [], "layout": _make_plotly_dark_layout()},
            ),
            html.Div(className="curve-stats", children=[
                html.Div(children=[
                    html.Div("PERIOD",   className="stat-chip-label"),
                    html.Div(id="curve-stats-period", className="stat-chip-value", children="—"),
                ]),
                html.Div(children=[
                    html.Div("DEPTH",    className="stat-chip-label"),
                    html.Div(id="curve-stats-depth",  className="stat-chip-value", children="—"),
                ]),
                html.Div(children=[
                    html.Div("BLS",      className="stat-chip-label"),
                    html.Div(id="curve-stats-bls",    className="stat-chip-value", children="—"),
                ]),
            ]),
        ]),

        # Panel: My Contributions (center-top)
        html.Div(className="panel panel-network", children=[
            html.Div(className="panel-scanline"),
            html.Div("MY CONTRIBUTIONS", className="panel-title"),
            html.Div(className="network-stat-grid", children=[
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="contrib-stars",      className="big-number gold",   children="0"),
                    html.Div("STARS ANALYZED",        className="big-number-label"),
                ]),
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="contrib-candidates", className="big-number green",  children="0"),
                    html.Div("CANDIDATES FOUND",      className="big-number-label"),
                ]),
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="contrib-uptime",     className="big-number cyan",   children="0h"),
                    html.Div("COMPUTE TIME",          className="big-number-label"),
                ]),
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="contrib-best",       className="big-number purple", children="—"),
                    html.Div("TOP SCORE",             className="big-number-label"),
                ]),
            ]),
        ]),

        # Panel: My Machine (right, spans 2 rows)
        html.Div(className="panel panel-nodes", children=[
            html.Div(className="panel-scanline"),
            html.Div("MY MACHINE", className="panel-title"),
            html.Div(id="my-machine-detail", children=[
                html.Div("CONNECTING...", className="empty-state"),
            ]),
        ]),

        # Panel: My Best Find (center-bottom)
        html.Div(className="panel panel-detail", children=[
            html.Div(className="panel-scanline"),
            html.Div(className="detail-header", children=[
                html.Span("MY BEST FIND", className="panel-title",
                          style={"border": "none", "padding": "0", "margin": "0", "flex": "1"}),
                html.Span(id="detail-title-badge", className="score-badge badge-candidate",
                          children="NO DATA"),
            ]),
            dcc.Graph(
                id="detail-curve-graph",
                config={"displayModeBar": False},
                style={"flex": "1", "minHeight": "0"},
                figure={"data": [], "layout": _make_plotly_dark_layout()},
            ),
        ]),

        # Panel: My Queue (bottom-left)
        html.Div(className="panel panel-queue", children=[
            html.Div(className="panel-scanline"),
            html.Div("MY QUEUE", className="panel-title"),
            html.Div(id="my-queue-content", className="queue-stat-row", children=[
                html.Div("CONNECTING...", className="empty-state"),
            ]),
        ]),

        # Panel: My History (bottom-center) — recent stars processed by this node
        html.Div(className="panel panel-history", children=[
            html.Div(className="panel-scanline"),
            html.Div("MY HISTORY", className="panel-title"),
            html.Div(id="my-history-content", className="history-list", children=[
                html.Div("NO HISTORY YET", className="empty-state"),
            ]),
        ]),

        # Panel: My Findings table (bottom-right)
        html.Div(className="panel panel-cands", children=[
            html.Div(className="panel-scanline"),
            html.Div("MY FINDINGS", className="panel-title"),
            dash_table.DataTable(
                id="candidates-table",
                columns=CANDIDATE_COLUMNS,
                data=[],
                style_table={
                    "overflowY": "auto",
                    "height": "100%",
                    "background": "transparent",
                },
                style_header=TABLE_STYLE_HEADER,
                style_cell=TABLE_STYLE_CELL,
                style_data_conditional=TABLE_STYLE_DATA_CONDITIONAL,
                page_action="none",
                sort_action="native",
                sort_by=[{"column_id": "reported_at", "direction": "desc"}],
            ),
        ]),
    ]),

    # Ticker bar
    html.Div(className="ticker-bar", children=[
        html.Div("LUMINA", className="ticker-label"),
        html.Div(className="ticker-track", children=[_make_ticker()]),
    ]),
])
