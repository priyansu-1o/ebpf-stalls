#!/usr/bin/env python3
import glob, os, sys, hashlib
from pathlib import Path

_site_pkg = Path(__file__).resolve().parent.parent / ".site-packages"
if _site_pkg.exists() and str(_site_pkg) not in sys.path:
    sys.path.insert(0, str(_site_pkg))

import pandas as pd

bad = 0
seen = {}
for d in sorted(glob.glob("data/tune/*") + glob.glob("data/final/*")):
    p = f"{d}/events.csv"
    if not os.path.exists(p):
        continue
    ev = pd.read_csv(p)
    r = (ev.ts_ns % 1000 == 0).mean()
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if r > 0.02 or len(ev) < 50 or h in seen:
        print("FAIL", d, "round", round(r, 3), "events", len(ev), "dup of", seen.get(h))
        bad += 1
    seen[h] = d
sys.exit(1 if bad else 0)

