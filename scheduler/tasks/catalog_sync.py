"""
Catalogue sync task: load NASA's lists of known objects into the API.

Downloads the Kepler KOI cumulative table, the TESS TOI table, the K2
planets-and-candidates table and the confirmed-planet table (for planets with
a TESS star id that were never TOIs) from the NASA Exoplanet Archive, and
replaces the API's known_objects catalogue (PUT /admin/catalog), which
relabels every candidate. Each object carries its period, a mid-transit time
(BJD_TDB) and duration, so a match needs the transit times to agree too.
"""
from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timezone

import httpx

from scheduler.config import get_settings

TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
# Order matters for names: on ties the first object wins, so a TESS planet
# that is also confirmed shows "HAT-P-8 b" rather than "TOI-...".
QUERIES = {
    "kepler": "select kepid,kepoi_name,kepler_name,koi_disposition,koi_period,koi_time0bk,koi_duration from cumulative",
    # Confirmed planets on TESS stars, including ones that never became TOIs (RV-first, ground surveys)
    "confirmed": "select tic_id,pl_name,pl_orbper,pl_tranmid,pl_trandur from pscomppars where tic_id is not null",
    "tess":   "select tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh from toi",
    "k2":     "select epic_hostname,pl_name,disposition,pl_orbper,pl_tranmid,pl_trandur from k2pandc",
}
STATUS = {
    "CONFIRMED": "known_planet", "CP": "known_planet", "KP": "known_planet",
    "CANDIDATE": "known_candidate", "PC": "known_candidate", "APC": "known_candidate",
    "FALSE POSITIVE": "known_false_positive", "FP": "known_false_positive",
    "FA": "known_false_positive", "REFUTED": "known_false_positive",
}
BKJD = 2454833.0   # Kepler time = BJD_TDB - 2454833


def _digits(s: str) -> int | None:
    d = "".join(ch for ch in (s or "") if ch.isdigit())
    return int(d) if d else None


def _num(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v == v else None   # drop NaN


def _pos(s: str) -> float | None:
    v = _num(s)
    return v if v is not None and v > 0 else None


def parse(source: str, text: str) -> list[dict]:
    """One catalogue object per row: {star, name, status, period, t0_bjd, duration_days}."""
    out = []
    for r in csv.DictReader(io.StringIO(text)):
        if source == "kepler":
            mission, star, disp = "kepler", _digits(r["kepid"]), r["koi_disposition"]
            name = r["kepler_name"] or "KOI-" + r["kepoi_name"].lstrip("K0")
            period, t0, dur_h = _pos(r["koi_period"]), _num(r["koi_time0bk"]), _pos(r["koi_duration"])
            t0 = t0 + BKJD if t0 is not None else None
        elif source == "tess":
            mission, star, disp = "tess", _digits(r["tid"]), r["tfopwg_disp"]
            name = f"TOI-{r['toi']}"
            period, t0, dur_h = _pos(r["pl_orbper"]), _num(r["pl_tranmid"]), _pos(r["pl_trandurh"])
        elif source == "k2":
            mission, star, disp = "k2", _digits(r["epic_hostname"]), r["disposition"]
            name = r["pl_name"]
            period, t0, dur_h = _pos(r["pl_orbper"]), _num(r["pl_tranmid"]), _pos(r["pl_trandur"])
        else:   # confirmed planets with a TESS star id
            mission, star, disp = "tess", _digits(r["tic_id"]), "CONFIRMED"
            name = r["pl_name"]
            period, t0, dur_h = _pos(r["pl_orbper"]), _num(r["pl_tranmid"]), _pos(r["pl_trandur"])
        status = STATUS.get((disp or "").strip().upper())
        if star is None or status is None:
            continue
        out.append({"star": f"{mission}:{star}", "name": (name or "")[:80], "status": status, "period": period,
                    "t0_bjd": t0, "duration_days": dur_h / 24 if dur_h else None})
    return out


async def run_catalog_sync() -> dict:
    settings = get_settings()
    started = datetime.now(timezone.utc)
    t0 = time.time()
    objects: list[dict] = []
    errors: list[str] = []
    async with httpx.AsyncClient(timeout=180.0) as tap:
        for source, query in QUERIES.items():
            try:
                r = await tap.get(TAP, params={"query": query, "format": "csv"})
                r.raise_for_status()
                objects += parse(source, r.text)
            except Exception as exc:   # one table failing must not wipe the others
                errors.append(f"{source}: {exc}")
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
