"""Standalone dev server — no MongoDB required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dash
from dash import Input, Output
from dashboard.layout import layout, _CSS

# Escape CSS for a JS template literal
_CSS_JS = _CSS.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")

app = dash.Dash(
    __name__,
    title="Lumina — My Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
)
app.layout = layout

# Inject CSS on first render — clientside callbacks are executed by the browser JS engine
app.clientside_callback(
    f"""function(n) {{
        if (!document.getElementById('lumina-inline-css')) {{
            var s = document.createElement('style');
            s.id = 'lumina-inline-css';
            s.textContent = `{_CSS_JS}`;
            document.head.appendChild(s);
        }}
        return n;
    }}""",
    Output("_css-injector", "data"),
    Input("interval-1s", "n_intervals"),
)

if __name__ == "__main__":
    print("Starting at http://127.0.0.1:8050")
    app.run(host="127.0.0.1", port=8050, debug=False, use_reloader=False)
