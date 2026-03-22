import dash

app = dash.Dash(
    __name__,
    title="Lumina Mission Control",
    update_title=None,
    suppress_callback_exceptions=True,
)
server = app.server

from dashboard.layout import layout
from dashboard import callbacks  # noqa: F401 — registers all @app.callback decorators

app.layout = layout


def run(config: dict) -> None:
    from dashboard.db import init_db
    init_db(config)
    app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)


if __name__ == "__main__":
    import json
    with open(r"C:\Program Files\Lumina\Data\config\config.json") as f:
        cfg = json.load(f)
    run(cfg)
