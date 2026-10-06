"""
Cross-match candidates against NASA's catalogues of known objects.

The scheduler loads the KOI, TOI and K2 tables into the known_objects
collection (PUT /admin/catalog). Every candidate is labelled:

  known_planet          period matches a confirmed / known planet
  known_candidate       matches a catalogued planet candidate (KOI, PC/APC TOI)
  known_false_positive  matches a catalogued false positive (e.g. eclipsing binary)
  new                   no catalogued object at this period

plus known_star: whether the star has any catalogued object at all (a new
period on a known planet host is a classic place to find another planet).
Only "new" candidates go forward for review; known planets are kept as a live
check that nodes still recover them.
"""
from __future__ import annotations

from api import database as db

PERIOD_TOL = 0.01                    # same rule as training's period-matched labels
_RANK = {"known_planet": 0, "known_candidate": 1, "known_false_positive": 2}


def period_matches(period: float, catalogued: float) -> bool:
    return any(abs(period - catalogued * m) <= PERIOD_TOL * catalogued * m for m in (1.0, 2.0, 0.5))


def star_key(mission: str, star: str | int) -> str:
    """Mission-qualified numeric star id: 'kepler:9941662', 'tess:261136679'."""
    digits = "".join(ch for ch in str(star) if ch.isdigit())
    return f"{mission.lower()}:{int(digits) if digits else 0}"


async def classify(mission: str, star: str, period: float) -> dict:
    objs = await db.known_objects().find({"star": star_key(mission, star)}, {"_id": 0}).to_list(length=200)
    hits = [o for o in objs if o.get("period") and period_matches(period, o["period"])]
    if not hits:
        return {"status": "new", "known_star": bool(objs), "name": None, "catalog_period": None}
    best = min(hits, key=lambda o: _RANK[o["status"]])
    return {"status": best["status"], "known_star": True, "name": best["name"], "catalog_period": best["period"]}


async def relabel_all() -> int:
    """Re-run the cross-match for every candidate (after a catalogue refresh)."""
    # ponytail: one pass over all candidates; batch by star if this gets slow.
    n = 0
    async for c in db.candidates().find({}, {"mission": 1, "tic_id": 1, "period_days": 1}):
        label = await classify(c["mission"], c["tic_id"], c["period_days"])
        await db.candidates().update_one({"_id": c["_id"]}, {"$set": {"catalog": label}})
        n += 1
    return n
