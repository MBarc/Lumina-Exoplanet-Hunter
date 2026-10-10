"""
Cross-match candidates against NASA's catalogues of known objects.

The scheduler loads the KOI, TOI, K2 and confirmed-planet tables into the
known_objects collection (PUT /admin/catalog). Every candidate is labelled:

  known_planet          same signal as a confirmed / known planet
  known_candidate       same signal as a catalogued planet candidate
  known_false_positive  same signal as a catalogued false positive
  new                   no catalogued object explains it
  unchecked             no catalogue loaded yet

plus known_star (the star has any catalogued object) and alias (it matches a
catalogued signal only at a multiple of its period). Only "new" goes forward
for review; known planets stay in the queue as a live check that nodes still
recover them.

"Same signal" needs the period AND the transit times to agree. Period alone
would label a genuinely new planet near a 2:1 resonance with a known one as
"known", which is the worst mistake this project can make.
"""
from __future__ import annotations

from pymongo import UpdateOne

from api import database as db

PERIOD_TOL = 0.01                  # same rule as training's period-matched labels
HARMONICS = (2.0, 0.5, 3.0, 1 / 3)  # aliases BLS commonly reports; epoch-checked only
_RANK = {"known_planet": 0, "known_candidate": 1, "known_false_positive": 2}
# Mission clocks are BJD_TDB minus a constant: Kepler/K2 BKJD, TESS BTJD.
BJD_OFFSET = {"kepler": 2454833.0, "k2": 2454833.0, "tess": 2457000.0}


def star_key(mission: str, star: str | int) -> str:
    """Mission-qualified numeric star id: 'kepler:9941662', 'tess:261136679'."""
    digits = "".join(ch for ch in str(star) if ch.isdigit())
    return f"{mission.lower()}:{int(digits) if digits else 0}"


def _close(p: float, q: float) -> bool:
    return abs(p - q) <= PERIOD_TOL * q


def _epoch_ok(t0: float | None, duration: float | None, obj: dict, step: float) -> bool | None:
    """Do the candidate's transit times line up with the catalogued ones?

    step = the spacing on which both transit trains must coincide (the shorter
    of the two periods). None when either side lacks an epoch.
    """
    if t0 is None or obj.get("t0_bjd") is None:
        return None
    off = (t0 - obj["t0_bjd"]) % step
    off = min(off, step - off)
    dur = max(duration or 0.0, obj.get("duration_days") or 0.0)
    # ponytail: fixed tolerance; propagate catalogue period uncertainty over the
    # baseline if long-baseline TESS-vs-Kepler matches start slipping.
    return off <= max(1.5 * dur, 0.1)


def _stays_aligned(period: float, cat_period: float, mult: float, duration: float | None,
                   n_transits: float | None, obj: dict) -> bool:
    """Two transit trains that line up once can still drift apart: over the
    candidate's observed transits the accumulated period mismatch must stay
    within the timing tolerance. Unknown transit count -> assume 20 (strict)."""
    n = n_transits if n_transits and n_transits > 0 else 20
    drift = n * abs(period - cat_period * mult)
    dur = max(duration or 0.0, obj.get("duration_days") or 0.0)
    return drift <= max(1.5 * dur, 0.1)


def label(objs: list[dict], period: float, t0_bjd: float | None, duration: float | None,
          n_transits: float | None = None) -> dict:
    """Pure cross-match of one signal against one star's catalogued objects."""
    best = None
    for o in objs:
        p = o.get("period")
        if not p:
            continue
        mult = next((h for h in (1.0, *HARMONICS) if _close(period, p * h)), None)
        if mult is None:
            continue
        alias = mult != 1.0
        ok = _epoch_ok(t0_bjd, duration, o, min(period, p))
        if alias:
            # A harmonic is the known object only if the transit times line up
            # AND stay lined up across the observed span.
            if not ok or not _stays_aligned(period, p, mult, duration, n_transits, o):
                continue
        elif ok is False or (ok and not _stays_aligned(period, p, 1.0, duration, n_transits, o)):
            # Same period but different or drifting transit times = a different
            # planet in the system; unknown epoch = trust the period.
            continue
        key = (_RANK[o["status"]], alias)
        if best is None or key < best[0]:
            best = (key, o, alias)
    if best is None:
        return {"status": "new", "known_star": bool(objs), "alias": False, "name": None, "catalog_period": None}
    _, o, alias = best
    return {"status": o["status"], "known_star": True, "alias": alias, "name": o["name"],
            "catalog_period": o["period"]}


def t0_bjd_of(mission: str, t0: float | None) -> float | None:
    off = BJD_OFFSET.get(mission.lower())
    return t0 + off if t0 is not None and off is not None else None


async def classify(mission: str, star: str, period: float, t0: float | None, duration: float | None,
                   n_transits: float | None = None) -> dict:
    if await db.known_objects().estimated_document_count() == 0:
        return {"status": "unchecked", "known_star": False, "alias": False, "name": None, "catalog_period": None}
    objs = await db.known_objects().find({"star": star_key(mission, star)}, {"_id": 0}).to_list(length=500)
    return label(objs, period, t0_bjd_of(mission, t0), duration, n_transits)


async def relabel_all() -> int:
    """Re-run the cross-match for every candidate after a catalogue refresh:
    catalogue in memory once, one bulk write of the labels that changed."""
    by_star: dict[str, list[dict]] = {}
    async for o in db.known_objects().find({}, {"_id": 0}):
        by_star.setdefault(o["star"], []).append(o)
    ops = []
    async for c in db.candidates().find({}, {"mission": 1, "tic_id": 1, "period_days": 1,
                                             "t0": 1, "duration_days": 1, "n_transits": 1, "catalog": 1}):
        new = label(by_star.get(star_key(c["mission"], c["tic_id"]), []), c["period_days"],
                    t0_bjd_of(c["mission"], c.get("t0")), c.get("duration_days"), c.get("n_transits"))
        if new != c.get("catalog"):
            ops.append(UpdateOne({"_id": c["_id"]}, {"$set": {"catalog": new}}))
    if ops:
        await db.candidates().bulk_write(ops, ordered=False)
    return len(ops)
