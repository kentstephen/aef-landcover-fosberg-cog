# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""The ignition head for the map: one logistic head on the 10 m pixel embedding (64 bands, unit length)
plus the hour's weather, case against the 20 to 30 km both-differ control, all causes. The weather
part is restricted to what the browser computes from the HRRR run (FFWI, or FFWI with T, RH and wind),
and the cost of that restriction is printed against the full weather set. Writes
models/ignition_logit.json in the land cover head's format (mean, scale, coef, intercept) with the
weather columns after the 64 bands, so the browser folds the standardization the same way.

    uv run ignition/export_head.py --tag big
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import FEATS, load  # noqa: E402

WEATHER = {
    "ffwi": ["ffwi"],
    "ffwi+t+rh+wind": ["ffwi", "t_c", "rh", "wind_ms"],
    "full": FEATS["W"],
}
OUT = Path(__file__).resolve().parent.parent / "models" / "ignition_logit.json"


def cv_auc(X, y, g):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="big")
    ap.add_argument("--ctrl", default="r30_ctrl")
    ap.add_argument("--weather", default="ffwi", choices=list(WEATHER))
    a = ap.parse_args()
    c = load(a.tag)
    d = c[c.kind.isin(["case", a.ctrl])].copy()
    y = (d.kind == "case").to_numpy(int)
    g = d.sq.to_numpy()
    P = FEATS["P"]
    print(f"{len(d)} rows, {y.sum()} cases vs {a.ctrl}; AUC held out by HRRR square")
    for name, cols in WEATHER.items():
        X = np.nan_to_num(d[cols].to_numpy(np.float64))
        Xp = np.nan_to_num(d[P + cols].to_numpy(np.float64))
        print(f"  {name:16s} W alone {cv_auc(X, y, g):.3f}   P+W {cv_auc(Xp, y, g):.3f}")
    for cause in ["Natural", "Human"]:
        dd = d[d.cause == cause]
        yy = (dd.kind == "case").to_numpy(int)
        Xp = np.nan_to_num(dd[P + WEATHER[a.weather]].to_numpy(np.float64))
        print(f"  {cause}: P+{a.weather} {cv_auc(Xp, yy, dd.sq.to_numpy()):.3f}")
    cols = P + WEATHER[a.weather]
    X = np.nan_to_num(d[cols].to_numpy(np.float64))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X, y)
    sc, lr = m[0], m[1]
    # the ground part alone, standardized: what the map draws with no weather
    zg = ((X[:, :64] - sc.mean_[:64]) / sc.scale_[:64]) @ lr.coef_[0, :64]
    out = {
        "kind": "logit", "teacher": f"wfigs ignitions vs {a.ctrl}, {a.tag} set, weather {a.weather}",
        "weather": WEATHER[a.weather],
        "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(),
        "ground_logit_p01": float(np.percentile(zg, 1)), "ground_logit_p99": float(np.percentile(zg, 99)),
        "n_cases": int(y.sum()), "n_controls": int(len(y) - y.sum()),
    }
    OUT.write_text(json.dumps(out))
    print(f"wrote {OUT}; ground logit p1 {out['ground_logit_p01']:.2f}, p99 {out['ground_logit_p99']:.2f}; "
          f"weather coef {dict(zip(WEATHER[a.weather], lr.coef_[0, 64:].round(3)))}, intercept {lr.intercept_[0]:.3f}")


if __name__ == "__main__":
    sys.exit(main())
