# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Does the weather's memory matter, and how much, at the same point: ignition hours against the same point's
other hours (t_ctrl), so the ground cancels and only the hour differs. Compares this hour's FFWI with the past
24 hours' maximum FFWI and the past 72 hours' rain, and prints the standardized weights and the odds change
for a typical hour-to-hour swing. Also, how big the hourly swing in the weather term is at one point against
the spread of the ground term across points: how much of the picture should move.

    uv run ignition/hour_memory.py --tag big
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import load  # noqa: E402

SETS = {
    "ffwi (this hour)": ["ffwi"],
    "max24 (past day's peak)": ["ffwi_max24"],
    "ffwi + max24": ["ffwi", "ffwi_max24"],
    "ffwi + precip72": ["ffwi", "precip72_mm"],
    "ffwi + max24 + precip72": ["ffwi", "ffwi_max24", "precip72_mm"],
    "ffwi + t + rh + wind": ["ffwi", "t_c", "rh", "wind_ms"],
    "ffwi + t + rh + wind + max24": ["ffwi", "t_c", "rh", "wind_ms", "ffwi_max24"],
    "wind alone": ["wind_ms"],
    "rh alone": ["rh"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="big")
    a = ap.parse_args()
    c = load(a.tag)
    d = c[c.kind.isin(["case", "t_ctrl"])].copy()
    d = d[d.ffwi_max24.notna()]
    y = (d.kind == "case").to_numpy(int)
    g = d.sq.to_numpy()
    print(f"{len(d)} rows: {y.sum()} ignition hours vs the same points' other hours; AUC held out by HRRR square")
    for name, cols in SETS.items():
        X = np.nan_to_num(d[cols].to_numpy(np.float64))
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pred[te] = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X, y)
        w = ", ".join(f"{col} {b:+.2f}" for col, b in zip(cols, m[1].coef_[0]))
        print(f"  {name:32s} AUC {roc_auc_score(y, pred):.3f}   weights per sd: {w}")
    # the hour's swing against the ground's spread
    cases = c[c.kind == "case"]
    print("\nHow much should move: at one point, this hour's FFWI against the same point's other hours")
    print(f"  FFWI at the ignition hour: median {cases.ffwi.median():.1f}, p90 {cases.ffwi.quantile(.9):.1f}; other hours at the same points: median {d[d.kind == 't_ctrl'].ffwi.median():.1f}, p90 {d[d.kind == 't_ctrl'].ffwi.quantile(.9):.1f}")
    print(f"  past-day peak at the ignition hour: median {cases.ffwi_max24.median():.1f}; at the other hours {d[d.kind == 't_ctrl'].ffwi_max24.median():.1f}")
    sd_hour = d.groupby("case_id").ffwi.std().median()
    print(f"  within one point, sd of FFWI between its hours (median over points): {sd_hour:.1f}; across points at the ignition hour: {cases.ffwi.std():.1f}")


if __name__ == "__main__":
    sys.exit(main())
