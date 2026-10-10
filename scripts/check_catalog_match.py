"""
Check the catalogue cross-match rules (api/catalog.label) on the cases that matter.

The worst failure is labelling a genuinely new planet as "known" and hiding it
from NEW, e.g. an outer planet near 2:1 resonance with a known one.

Run inside the API environment:  python -m scripts.check_catalog_match
"""
from api.catalog import label

P, T0, DUR = 1.763588, 2455014.5, 3.0 / 24          # Kepler-13 b-like ephemeris (BJD)
k13 = [{"name": "Kepler-13 b", "status": "known_planet", "period": P, "t0_bjd": T0, "duration_days": DUR}]

cases = [
    ("exact period, aligned transit", dict(period=1.7637, t0_bjd=T0 + 500 * P, duration=DUR), "known_planet", False),
    ("2x period alias, aligned", dict(period=2 * P, t0_bjd=T0 + 300 * P, duration=DUR), "known_planet", True),
    ("3x period alias, aligned", dict(period=3 * P, t0_bjd=T0 + 30 * P, duration=DUR), "known_planet", True),
    ("half period alias, aligned", dict(period=P / 2, t0_bjd=T0 + 7.5 * P, duration=DUR), "known_planet", True),
    ("NEW planet near 2:1, other transit times", dict(period=3.55, t0_bjd=T0 + 0.6, duration=DUR), "new", False),
    ("same period, clearly different transit times", dict(period=P, t0_bjd=T0 + 0.5 * P, duration=DUR), "new", False),
    ("exact period, no transit time recorded", dict(period=P, t0_bjd=None, duration=DUR), "known_planet", False),
    ("harmonic but no transit time: not assumed known", dict(period=2 * P, t0_bjd=None, duration=DUR), "new", False),
]
for name, kw, status, alias in cases:
    got = label(k13, kw["period"], kw["t0_bjd"], kw["duration"])
    assert (got["status"], got["alias"]) == (status, alias), f"{name}: {got}"
    if status == "new":
        assert got["known_star"], name
    print(f"ok  {name:<48} -> {got['status']}{' (alias)' if got['alias'] else ''}")

# Astra's case: transits coincide once, then drift (20.18 d vs 2 x 10 d).
p10 = [{"name": "Known-10d b", "status": "known_planet", "period": 10.0, "t0_bjd": T0, "duration_days": 0.2}]
drift = label(p10, 20.18, T0, 0.2, n_transits=40)
assert drift["status"] == "new", f"drifting 2:1 neighbour must stay new: {drift}"
assert label(p10, 20.0, T0 + 30 * 20.0, 0.2, n_transits=40)["status"] == "known_planet"
print("ok  drifting near-2:1 neighbour stays new; exact 2x alias is known")

fp_and_planet = k13 + [{"name": "KOI-13.02", "status": "known_false_positive", "period": P,
                        "t0_bjd": T0, "duration_days": DUR}]
assert label(fp_and_planet, P, T0, DUR)["status"] == "known_planet", "planet must outrank FP"
assert label([], 2.0, T0, DUR) == {"status": "new", "known_star": False, "alias": False,
                                   "name": None, "catalog_period": None}
print("ok  precedence and unknown star")
