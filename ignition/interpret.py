# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""What the ground head reads, and whether fire size is predictable.

1. Within one NLCD class, can the embedding still tell case points from the same-hour control points
   2.5 km away? That is the part of the ground signal NLCD cannot carry. Out-of-fold ground head
   (E, case vs s_ctrl), AUC per NLCD class of the point, with counts.
2. The ground head's out-of-fold score by NLCD class: what it lights.
3. Size: among cases, is the size (WFIGS IncidentSize; FinalAcres is set for only 12% of fires) above the
   90th percentile, or at least 100 acres, readable from W, E, N (AUC, GroupKFold)?

    uv run ignition/interpret.py                       # the pilot set, 2.5 km control
    uv run ignition/interpret.py --tag big --ctrl s30  # the big set, the 20 to 30 km ring
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import CLASSES, FEATS, load_set  # noqa: E402

warnings.filterwarnings("ignore")
NAMES = {11: "water", 12: "snow", 21: "dev open", 22: "dev low", 23: "dev med", 24: "dev high", 31: "barren", 41: "decid",
         42: "evergrn", 43: "mixed", 52: "shrub", 71: "grass", 81: "pasture", 82: "crops", 90: "wood wet", 95: "herb wet"}


def oof(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--ctrl", default="s", help="spatial control ring: s (2.5 km), s10, s30")
    ap.add_argument("--size-col", default="size", help="size (IncidentSize) or final_acres")
    args = ap.parse_args()
    ctrl = f"{args.ctrl}_ctrl"
    c = load_set(args.tag)
    for cl in CLASSES:
        c[f"nlcd_is{cl}"] = (c.nlcd == cl).astype(float)
    c["sq"] = c.cy * 100 + c.cx
    for cause in ["all", "Natural", "Human"]:
        d = c if cause == "all" else c[c.cause == cause]
        d = d[d.kind.isin(["case", ctrl])].copy()
        X = np.nan_to_num(d[FEATS["E"]].to_numpy(np.float64))
        y = (d.kind == "case").to_numpy(int)
        d["g"] = oof(X, y, d.sq.to_numpy())
        print(f"\n== {cause}: ground head (E, case vs {ctrl}), overall AUC {roc_auc_score(y, d.g):.3f} ==")
        print(f"{'NLCD class':12s} {'n case':>7s} {'n ctrl':>7s} {'AUC within':>11s} {'mean score':>11s}")
        for cl, g in d.groupby("nlcd"):
            yy = (g.kind == "case").to_numpy(int)
            a = roc_auc_score(yy, g.g) if 0 < yy.sum() < len(yy) and len(yy) >= 20 else np.nan
            print(f"{NAMES.get(cl, str(cl)):12s} {yy.sum():7d} {len(yy) - yy.sum():7d} {a:11.3f} {g.g.mean():11.3f}")
        # the within-class pooled number: AUC after removing each class's mean score, which is what NLCD could know
        d["g_res"] = d.g - d.groupby("nlcd").g.transform("mean")
        print(f"pooled within-class AUC (score minus the class mean): {roc_auc_score(y, d.g_res):.3f}")

    sz = args.size_col
    cs = c[c.kind == "case"].copy()
    cs = cs[cs[sz].notna() & (cs[sz] > 0)]
    for label, thr_of in [("above the 90th percentile", lambda d: d[sz].quantile(0.9)), ("at least 100 acres", lambda d: 100.0)]:
        print(f"\n== size: {sz} {label} among cases, AUC held out by HRRR square ==")
        for cause in ["all", "Natural", "Human"]:
            d = cs if cause == "all" else cs[cs.cause == cause]
            thr = thr_of(d)
            y = (d[sz] >= thr).to_numpy(int)
            if y.sum() < 20 or y.sum() > len(y) - 20:
                print(f"{cause:8s} n {len(d)}, {y.sum()} positives: too few")
                continue
            out = []
            for name in ["W", "E", "N", "E+W", "N+W", "E+N+W"]:
                cols = sum((FEATS[p] for p in name.split("+")), [])
                X = np.nan_to_num(d[cols].to_numpy(np.float64))
                out.append(f"{name} {roc_auc_score(y, oof(X, y, d.sq.to_numpy())):.3f}")
            print(f"{cause:8s} n {len(d)}, {y.sum()} positives, threshold {thr:.0f} acres: " + ", ".join(out))

if __name__ == "__main__":
    sys.exit(main())
