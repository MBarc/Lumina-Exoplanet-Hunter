from dash import html, dcc, dash_table

CANDIDATE_COLUMNS = [
    {"name": "TIC ID",      "id": "tic_id"},
    {"name": "Sector",      "id": "sector"},
    {"name": "Period",      "id": "period_days"},
    {"name": "Duration",    "id": "duration_days"},
    {"name": "Depth",       "id": "depth_ppm"},
    {"name": "BLS Power",   "id": "bls_power"},
    {"name": "Score",       "id": "exonet_score"},
    {"name": "Label",       "id": "label"},
    {"name": "Node",        "id": "worker_hostname"},
    {"name": "Reported",    "id": "reported_at"},
]

TABLE_STYLE_HEADER = {
    "backgroundColor": "#0d1f3c",
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
    },
    {
        "if": {"filter_query": "{label} = FALSE_POSITIVE"},
        "color": "#4a6380",
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
            gridcolor="rgba(255,255,255,0.05)",
            gridwidth=1,
            zeroline=False,
            showticklabels=False,
            color="#4a6380",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            gridwidth=1,
            zeroline=False,
            showticklabels=False,
            color="#4a6380",
        ),
    )
    base.update(kwargs)
    return base


layout = html.Div(className="app-container", children=[
    # Intervals
    dcc.Interval(id="interval-1s",  interval=1_000,  n_intervals=0),
    dcc.Interval(id="interval-5s",  interval=5_000,  n_intervals=0),
    dcc.Interval(id="interval-10s", interval=10_000, n_intervals=0),

    # Header
    html.Div(className="header", children=[
        html.Div(className="header-left", children=[
            html.Span("\u25c8", style={"color": "#00c8ff", "fontSize": "18px", "textShadow": "0 0 12px #00c8ff"}),
            html.Span("LUMINA MISSION CONTROL", className="header-title"),
        ]),
        html.Div(className="header-right", children=[
            html.Span(id="header-clock", className="utc-clock"),
            html.Span(id="node-count-badge", className="node-badge", children="0 NODES"),
        ]),
    ]),

    # Main grid
    html.Div(className="main-grid", children=[

        # Panel: Live Analysis (left, 2 rows)
        html.Div(className="panel panel-live", children=[
            html.Div("LIVE ANALYSIS", className="panel-title"),
            html.Div(id="live-tic-label", className="tic-label", children="AWAITING DATA"),
            dcc.Graph(
                id="live-curve-graph",
                config={"displayModeBar": False},
                style={"flex": "1", "minHeight": "0"},
                figure={
                    "data": [],
                    "layout": _make_plotly_dark_layout(),
                },
            ),
            html.Div(className="curve-stats", children=[
                html.Div(children=[
                    html.Div("PERIOD", className="stat-chip-label"),
                    html.Div(id="curve-stats-period", className="stat-chip-value", children="\u2014"),
                ]),
                html.Div(children=[
                    html.Div("DEPTH", className="stat-chip-label"),
                    html.Div(id="curve-stats-depth", className="stat-chip-value", children="\u2014"),
                ]),
                html.Div(children=[
                    html.Div("BLS", className="stat-chip-label"),
                    html.Div(id="curve-stats-bls", className="stat-chip-value", children="\u2014"),
                ]),
            ]),
        ]),

        # Panel: Network Stats (center-top)
        html.Div(className="panel panel-network", children=[
            html.Div("EXONET NETWORK", className="panel-title"),
            html.Div(className="network-stat-grid", children=[
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="network-nodes",      className="big-number gold",   children="0"),
                    html.Div("ACTIVE NODES",          className="big-number-label"),
                ]),
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="network-stars",      className="big-number cyan",   children="0"),
                    html.Div("STARS ANALYZED",        className="big-number-label"),
                ]),
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="network-candidates", className="big-number green",  children="0"),
                    html.Div("CANDIDATES FOUND",      className="big-number-label"),
                ]),
                html.Div(className="network-stat-cell", children=[
                    html.Div(id="network-queued",     className="big-number purple", children="0"),
                    html.Div("SECTORS QUEUED",        className="big-number-label"),
                ]),
            ]),
        ]),

        # Panel: Node Status (right, 2 rows)
        html.Div(className="panel panel-nodes", children=[
            html.Div("NODE STATUS", className="panel-title"),
            html.Div(id="node-list", children=[
                html.Div("NO ACTIVE NODES", className="empty-state"),
            ]),
        ]),

        # Panel: Transit Detail (center-bottom)
        html.Div(className="panel panel-detail", children=[
            html.Div(className="detail-header", children=[
                html.Span("TRANSIT SIGNATURE", className="panel-title",
                          style={"border": "none", "padding": "0", "margin": "0"}),
                html.Span(id="detail-title-badge", className="score-badge badge-candidate",
                          children="NO DATA"),
            ]),
            dcc.Graph(
                id="detail-curve-graph",
                config={"displayModeBar": False},
                style={"flex": "1", "minHeight": "0"},
                figure={
                    "data": [],
                    "layout": _make_plotly_dark_layout(),
                },
            ),
        ]),

        # Panel: Candidates Table (bottom, full width)
        html.Div(className="panel panel-cands", children=[
            html.Div("RECENT TRANSIT CANDIDATES", className="panel-title"),
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
])
