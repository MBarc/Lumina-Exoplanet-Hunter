"""
Does the shipped preprocessing recover known TESS planets?

Takes N confirmed-planet TOIs (TFOPWG "CP") with periods short enough for one
27-day sector, downloads one SPOC 2-minute light curve each from MAST, runs
ml.preprocess.preprocess (the code nodes and training use) and reports how
often a returned BLS candidate matches the catalogued period (1x / 2x / 0.5x,
1% tolerance — the same rule as training's period-matched labels).

Run:  python -m scripts.verify_bls_tess [N]
"""
from __future__ import annotations

import csv
import io
import sys
import tempfile
import time

import requests
from astroquery.mast import Observations

from ml.preprocess import preprocess
from ml.train import _matches_koi_period

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
       "select+tid,toi,pl_orbper,pl_trandep+from+toi+where+tfopwg_disp='CP'"
       "+and+pl_orbper+between+1+and+10+order+by+toi&format=csv")

rows = list(csv.DictReader(io.StringIO(requests.get(TAP, timeout=120).text)))
seen, targets = set(), []
for r in rows:
    if r["tid"] not in seen:
        seen.add(r["tid"])
        targets.append(r)
targets = targets[:N]

top1 = anyk = done = 0
t0 = time.time()
with tempfile.TemporaryDirectory() as tmp:
    for r in targets:
        tic, per = int(r["tid"]), float(r["pl_orbper"])
        try:
            obs = Observations.query_criteria(obs_collection="TESS", target_name=str(tic),
                                              dataproduct_type="timeseries", project="TESS")
            prods = Observations.filter_products(Observations.get_product_list(obs[:1]),
                                                 productSubGroupDescription="LC", extension="fits")
            path = Observations.download_products(prods[:1], download_dir=tmp, cache=False)["Local Path"][0]
            cands = preprocess(path)
        except Exception as exc:
            print(f"TOI {r['toi']:>8}  TIC {tic:<11} P={per:7.3f}  ERROR {type(exc).__name__}: {exc}")
            continue
        done += 1
        hit1 = bool(cands) and _matches_koi_period(float(cands[0].period), [per])
        hitk = any(_matches_koi_period(float(c.period), [per]) for c in cands)
        top1 += hit1
        anyk += hitk
        found = ", ".join(f"{c.period:.3f}" for c in cands)
        print(f"TOI {r['toi']:>8}  TIC {tic:<11} P={per:7.3f}  top={'Y' if hit1 else '-'} "
              f"any={'Y' if hitk else '-'}  BLS: {found}")

print(f"\n{done}/{len(targets)} processed in {time.time() - t0:.0f}s:  "
      f"top candidate matches {top1}/{done},  any candidate matches {anyk}/{done}")
