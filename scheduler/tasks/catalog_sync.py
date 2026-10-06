"""
Catalogue sync task: load NASA's lists of known objects into the API.

Downloads the Kepler KOI cumulative table, the TESS TOI table and the K2
planets-and-candidates table from the NASA Exoplanet Archive and replaces the
API's known_objects catalogue (PUT /admin/catalog), which relabels every
candidate as known planet / known candidate / known false positive / new.
"""
from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone

import httpx

from scheduler.config import get_settings

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
QUERIES = {
    "kepler": "select kepid,kepoi_name,kepler_name,koi_disposition,koi_period from cumulative",
    "tess":   "select tid,toi,tfopwg_disp,pl_orbper from toi",
    "k2":     "select epic_hostname,pl_name,disposition,pl_orbper from k2pandc",
}
STATUS = {
    "CONFIRMED": "known_planet", "CP": "known_planet", "KP": "known_planet",
    "CANDIDATE": "known_candidate", "PC": "known_candidate", "APC": "known_candidate",
    "FALSE POSITIVE": "known_false_positive", "FP": "known_false_positive",
    "FA": "known_false_positive", "REFUTED": "known_false_positive",
}


def _digits(s: str) -> int | None:
    d = "".join(ch for ch in s if ch.isdigit())
    return int(d) if d else None


def _period(s: str) -> float | None:
    try:
        p = float(s)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def parse(mission: str, text: str) -> list[dict]:
    """One catalogue object per row: {star, name, status, period}."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        if mission == "kepler":
            star, disp, per = _digits(r["kepid"]), r["koi_disposition"], r["koi_period"]
            name = r["kepler_name"] or "KOI-" + r["kepoi_name"].lstrip("K0")
        elif mission == "tess":
            star, disp, per = _digits(r["tid"]), r["tfopwg_disp"], r["pl_orbper"]
            name = f"TOI-{r['toi']}"
        else:
            star, disp, per = _digits(r["epic_hostname"]), r["disposition"], r["pl_orbper"]
            name = r["pl_name"]
        status = STATUS.get((disp or "").strip().upper())
        if star is None or status is None:
            continue
        out.append({"star": f"{mission}:{star}", "name": name[:80], "status": status, "period": _period(per)})
    return out


async def run_catalog_sync() -> dict:
    settings = get_settings()
    started = datetime.now(timezone.utc)
    t0 = time.time()
    objects: list[dict] = []
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=180.0) as tap:
        for mission, query in QUERIES.items():
            try:
                r = await tap.get(TAP, params={"query": query, "format": "csv"})
                r.raise_for_status()
                objects += parse(mission, r.text)
            except Exception as exc:   # one table failing must not wipe the others
                errors.append(f"{mission}: {exc}")
    if errors:
        # Never replace the catalogue with a partial one.
        return {"task": "catalog_sync", "started_at": started.isoformat(),
                "elapsed_s": round(time.time() - t0, 1), "errors": errors}
    async with httpx.AsyncClient(base_url=settings.api_url, timeout=600.0,
                                 headers={"X-API-Key": settings.api_key}) as api:
        r = await api.put("/admin/catalog", json={"source": f"NASA Exoplanet Archive {started:%Y-%m-%d}",
                                                  "objects": objects})
        r.raise_for_status()
        result = r.json()
    return {"task": "catalog_sync", "started_at": started.isoformat(),
            "elapsed_s": round(time.time() - t0, 1), "errors": [],
            "inserted": result["objects"], "done": result["candidates_relabelled"]}
