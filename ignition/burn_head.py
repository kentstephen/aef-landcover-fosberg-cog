# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
# ///
"""How long and how big a fire that starts here gets: the burn side of the ignition study.

The ignition cases (data/ignition/cases_<tag>.parquet, WFIGS fires with their 10 m embedding, the 150 m mean,
the prior-year NLCD class and the hour's weather) are joined back to WFIGS for FireOutDateTime, so each case
has a duration (discovery to out, hours) and a size (IncidentSize, acres). Then:
  1. duration and size by NLCD class at the ignition point (quantiles, the share still burning after a day,
     three days, a week): the hold hours per class for the map, from data instead of the first-cut table;
  2. ridge heads from the embedding (the pixel, the 150 m mean), the class dummies and the weather to
     log duration and log size, held out by HRRR square: does the embedding say more about burn than the class;
  3. logistic heads for "still burning after 24 h" and "100 acres or more".
Writes data/ignition/burn.txt (the report) and models/burn_duration.json (the hold hours per class and the
pixel-embedding ridge heads in the land cover head's format).

    uv run ignition/burn_head.py --tag big
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import CLASSES, D, FEATS, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NAMES = {11: "open water", 12: "snow and ice", 21: "developed open", 22: "developed low", 23: "developed medium",
         24: "developed high", 31: "barren", 41: "deciduous forest", 42: "evergreen forest", 43: "mixed forest",
         52: "shrub", 71: "grassland", 81: "pasture", 82: "crops", 90: "woody wetland", 95: "herbaceous wetland"}
MAX_HOURS = 24 * 365
HOLD_CAP = 96  # the texture holds a byte of half hours (HOLD_SCALE 2 in web/src/aef/fuel.ts); four days is its longest hold
out_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out_lines.append(s)


def cv_pred(model, X, y, g):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pred[te] = model().fit(X[tr], y[tr]).predict(X[te])
    return pred


def cv_proba(model, X, y, g):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pred[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pred


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="big")
    a = ap.parse_args()
    c = load(a.tag)
    c = c[c.kind == "case"].copy()
    say(f"{len(c)} ignition cases with embedding, class and weather ({a.tag})")

    # back to WFIGS for the out time: the point, the name and the year identify the fire
    w = pd.read_parquet(D / "wfigs.parquet", columns=["IncidentName", "FireDiscoveryDateTime", "FireOutDateTime", "IncidentSize", "geom_lon", "geom_lat"])
    w["year"] = w.FireDiscoveryDateTime.dt.year
    w["k"] = w.geom_lon.round(5).astype(str) + "|" + w.geom_lat.round(5).astype(str) + "|" + w.IncidentName.fillna("") + "|" + w.year.astype(str)
    w = w.drop_duplicates("k").set_index("k")
    c["k"] = c.lon.round(5).astype(str) + "|" + c.lat.round(5).astype(str) + "|" + c.name.fillna("") + "|" + c.year.astype(str)
    c = c.join(w[["FireDiscoveryDateTime", "FireOutDateTime"]], on="k")
    hrs = (c.FireOutDateTime - c.FireDiscoveryDateTime).dt.total_seconds() / 3600
    c["dur_h"] = hrs.where((hrs > 0) & (hrs <= MAX_HOURS))
    c["size_ac"] = c["size"].where(c["size"] > 0)
    say(f"joined {c.FireDiscoveryDateTime.notna().sum()}; with an out time {c.dur_h.notna().sum()}; with a size {c.size_ac.notna().sum()}")
    say(f"duration hours, all: p10 {c.dur_h.quantile(.1):.1f} p25 {c.dur_h.quantile(.25):.1f} p50 {c.dur_h.quantile(.5):.1f} "
        f"p75 {c.dur_h.quantile(.75):.1f} p90 {c.dur_h.quantile(.9):.1f}; still burning after 24 h {(c.dur_h > 24).mean() / c.dur_h.notna().mean():.2f}")

    # 1. by class at the ignition point
    say("\nBy NLCD class at the ignition point (duration = discovery to out, hours; size = IncidentSize, acres)")
    say(f"{'class':20s} {'n_dur':>6s} {'p25':>6s} {'p50':>6s} {'p75':>6s} {'>24h':>5s} {'>72h':>5s} {'>7d':>5s} | {'n_size':>6s} {'p50ac':>7s} {'p90ac':>8s} {'>=100ac':>7s}")
    hold = {}
    rows = {}
    for cl in CLASSES:
        d = c[c.nlcd == cl]
        dd = d.dur_h.dropna()
        ss = d.size_ac.dropna()
        if len(dd) < 20:
            say(f"{NAMES[cl]:20s} {len(dd):6d}  (too few)")
            continue
        q = dd.quantile([.25, .5, .75])
        rows[cl] = dict(n=int(len(dd)), p25=float(q[.25]), p50=float(q[.5]), p75=float(q[.75]),
                        over24=float((dd > 24).mean()), over72=float((dd > 72).mean()), over168=float((dd > 168).mean()),
                        n_size=int(len(ss)), size_p50=float(ss.median()) if len(ss) else None, size_p90=float(ss.quantile(.9)) if len(ss) else None,
                        size_ge100=float((ss >= 100).mean()) if len(ss) else None)
        r = rows[cl]
        say(f"{NAMES[cl]:20s} {r['n']:6d} {r['p25']:6.1f} {r['p50']:6.1f} {r['p75']:6.1f} {r['over24']:5.2f} {r['over72']:5.2f} {r['over168']:5.2f} | "
            f"{r['n_size']:6d} {r['size_p50'] or 0:7.2f} {r['size_p90'] or 0:8.1f} {r['size_ge100'] or 0:7.3f}")
    # the hold: the class's median hours from discovery to out, capped at two days; water and ice hold nothing
    for cl, r in rows.items():
        hold[cl] = 0 if cl in (11, 12) else int(round(min(HOLD_CAP, r["p50"])))
    say("\nHold hours per class (median hours to out, capped at %d; water and ice 0):" % HOLD_CAP)
    say("  " + ", ".join(f"{cl}: {h}" for cl, h in hold.items()))

    # 2. heads to log duration and log size
    P, E, N, W = FEATS["P"], FEATS["E"], FEATS["N"], ["ffwi", "t_c", "rh", "wind_ms"]
    sets = {"N (class)": N, "W": W, "N+W": N + W, "P (pixel)": P, "E (150 m)": E, "P+W": P + W, "E+W": E + W, "P+N+W": P + N + W, "E+N+W": E + N + W}
    heads = {}
    for target, col, tf in [("log duration (h)", "dur_h", np.log1p), ("log10 size (acres)", "size_ac", np.log10)]:
        d = c[c[col].notna()]
        y = tf(d[col].to_numpy(np.float64))
        g = d.sq.to_numpy()
        say(f"\n{target}: {len(d)} fires, ridge (alpha 10 on standardized columns), 5-fold by HRRR square")
        say(f"  {'features':12s} {'R2':>6s} {'rho':>6s}")
        for name, cols in sets.items():
            X = np.nan_to_num(d[cols].to_numpy(np.float64))
            p = cv_pred(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), X, y, g)
            say(f"  {name:12s} {r2(y, p):6.3f} {spearmanr(y, p).statistic:6.3f}")
        X = np.nan_to_num(d[P + W].to_numpy(np.float64))
        m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)
        heads[col] = {"kind": "ridge", "target": target, "weather": W, "mean": m[0].mean_.tolist(), "scale": m[0].scale_.tolist(),
                      "coef": m[1].coef_.tolist(), "intercept": float(m[1].intercept_), "n": int(len(d))}

    # 3. the two yes/no questions
    for label, col, thr in [("still burning after 24 h", "dur_h", 24), ("100 acres or more", "size_ac", 100)]:
        d = c[c[col].notna()]
        y = (d[col] >= thr).to_numpy(int)
        g = d.sq.to_numpy()
        say(f"\n{label}: {len(d)} fires, {y.mean():.2f} yes; logistic (C 0.3), AUC by HRRR square")
        for name in ["N (class)", "W", "N+W", "P (pixel)", "E (150 m)", "P+W", "E+W", "E+N+W"]:
            X = np.nan_to_num(d[sets[name]].to_numpy(np.float64))
            p = cv_proba(lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)), X, y, g)
            say(f"  {name:12s} {roc_auc_score(y, p):.3f}")

    (D / "burn.txt").write_text("\n".join(out_lines) + "\n")
    (ROOT / "models" / "burn_duration.json").write_text(json.dumps({
        "source": f"WFIGS ignition cases ({a.tag}), discovery to FireOutDateTime, IncidentSize; NLCD class at the point",
        "hold_hours_nlcd": hold, "hold_cap_hours": HOLD_CAP, "by_class": {str(k): v for k, v in rows.items()}, "heads": heads}, indent=1))
    say(f"\nwrote {D / 'burn.txt'} and models/burn_duration.json")


if __name__ == "__main__":
    sys.exit(main())
