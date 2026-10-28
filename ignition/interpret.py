# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""What the ground head reads, and whether fire size is predictable.

1. Within one NLCD class, can the embedding still tell case points from the same-hour control points
   2.5 km away? That is the part of the ground signal NLCD cannot carry. Out-of-fold ground head
   (E, case vs s_ctrl), AUC per NLCD class of the point, with counts.
2. The ground head's out-of-fold score by NLCD class: what it lights.
3. Size: among cases, is final size above the 90th percentile readable from W, E, N (AUC, GroupKFold)?

    uv run ignition/interpret.py
"""
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
    c = load_set()
    for cl in CLASSES:
        c[f"nlcd_is{cl}"] = (c.nlcd == cl).astype(float)
    c["sq"] = c.cy * 100 + c.cx
    for cause in ["all", "Natural", "Human"]:
        d = c if cause == "all" else c[c.cause == cause]
        d = d[d.kind.isin(["case", "s_ctrl"])].copy()
        X = np.nan_to_num(d[FEATS["E"]].to_numpy(np.float64))
        y = (d.kind == "case").to_numpy(int)
        d["g"] = oof(X, y, d.sq.to_numpy())
        print(f"\n== {cause}: ground head (E, case vs s_ctrl), overall AUC {roc_auc_score(y, d.g):.3f} ==")
        print(f"{'NLCD class':12s} {'n case':>7s} {'n ctrl':>7s} {'AUC within':>11s} {'mean score':>11s}")
        for cl, g in d.groupby("nlcd"):
            yy = (g.kind == "case").to_numpy(int)
            a = roc_auc_score(yy, g.g) if 0 < yy.sum() < len(yy) and len(yy) >= 20 else np.nan
            print(f"{NAMES.get(cl, str(cl)):12s} {yy.sum():7d} {len(yy) - yy.sum():7d} {a:11.3f} {g.g.mean():11.3f}")
        # the within-class pooled number: AUC after removing each class's mean score, which is what NLCD could know
        d["g_res"] = d.g - d.groupby("nlcd").g.transform("mean")
        print(f"pooled within-class AUC (score minus the class mean): {roc_auc_score(y, d.g_res):.3f}")

    print("\n== size: final acres above the 90th percentile among cases, AUC held out by HRRR square ==")
    cs = c[c.kind == "case"].copy()
    cs = cs[cs.final_acres.notna() & (cs.final_acres > 0)]
    for cause in ["all", "Natural", "Human"]:
        d = cs if cause == "all" else cs[cs.cause == cause]
        thr = d.final_acres.quantile(0.9)
        y = (d.final_acres >= thr).to_numpy(int)
        out = []
        for name in ["W", "E", "N", "E+W", "N+W", "E+N+W"]:
            cols = sum((FEATS[p] for p in name.split("+")), [])
            X = np.nan_to_num(d[cols].to_numpy(np.float64))
            out.append(f"{name} {roc_auc_score(y, oof(X, y, d.sq.to_numpy())):.3f}")
        print(f"{cause:8s} n {len(d)}, threshold {thr:.0f} acres: " + ", ".join(out))


if __name__ == "__main__":
    sys.exit(main())
