# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Does the ground, read from the embeddings, say where fires start? Tiny logistic heads on the
pilot cases, each case against each kind of control:

  case vs t_ctrl   same point, another hour: only the weather differs (the ground cannot help; AUC 0.5
                   for ground features is the check that nothing leaks)
  case vs s_ctrl   same hour, a point up to 2.5 km away: only the ground differs (the weather cannot help)
  case vs r_ctrl   another point and another hour: both differ

Feature sets: W weather (FFWI sustained and gust, prior-24 h max, 72 h precipitation, T, RH, wind),
E embedding (the 150 m unit mean, 64 bands, standardized), P the single 10 m pixel, N the NLCD
baseline (pixel class one-hot + 5 x 5 class fractions), and the sums. Held out by HRRR square
(GroupKFold, 5 folds), so nearby fires never sit on both sides. Reports AUC per cause.

    uv run ignition/fit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

D = Path(__file__).resolve().parent.parent / "data" / "ignition"
CLASSES = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]


def dequant(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def load():
    c = pd.read_parquet(D / "cases.parquet")
    w = pd.read_parquet(D / "weather.parquet")
    e = pd.read_parquet(D / "embeddings.parquet")
    n = pd.read_parquet(D / "nlcd.parquet")
    c = c.join(w, how="inner")
    # the ground features belong to the point: t_ctrl shares the case's, r_ctrl shares s_ctrl's
    key = c.case_id.astype(str) + "_" + c.kind.map({"case": "a", "t_ctrl": "a", "s_ctrl": "b", "r_ctrl": "b"})
    pt = c[c.kind.isin(["case", "s_ctrl"])].index
    e = e.loc[e.index.intersection(pt)]
    n = n.loc[n.index.intersection(pt)]
    E = np.stack([np.frombuffer(b, np.float32) for b in e.win_mean])
    P = dequant(np.stack([np.frombuffer(b, np.int8) for b in e.pix]))
    emb = pd.DataFrame(np.hstack([E, P]), index=e.index, columns=[f"e{i}" for i in range(64)] + [f"p{i}" for i in range(64)])
    emb["win_norm"] = e.win_norm.values
    emb["nodata"] = e.nodata.values
    ground = emb.join(n, how="inner")
    ground["pkey"] = key.loc[ground.index]
    c["pkey"] = key
    c = c.merge(ground, on="pkey", how="inner", suffixes=("", "_g"))
    c = c[~c.nodata & c.ffwi.notna() & (c.nlcd != 250)]
    for cl in CLASSES:
        c[f"nlcd_is{cl}"] = (c.nlcd == cl).astype(float)
    c["sq"] = c.cy * 100 + c.cx
    return c


FEATS = {
    "W": ["ffwi", "ffwi_gust", "ffwi_max24", "precip72_mm", "t_c", "rh", "wind_ms"],
    "E": [f"e{i}" for i in range(64)] + ["win_norm"],
    "P": [f"p{i}" for i in range(64)],
    "N": [f"nlcd_is{cl}" for cl in CLASSES] + [f"nlcd_f{cl}" for cl in CLASSES],
}
SETS = {"W": ["W"], "N": ["N"], "E": ["E"], "P": ["P"], "E+P": ["E", "P"], "N+W": ["N", "W"], "E+W": ["E", "W"], "E+N": ["E", "N"], "E+N+W": ["E", "N", "W"]}


def auc_cv(X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pred)


def main():
    c = load()
    print(f"rows {len(c)}: " + ", ".join(f"{k} {v}" for k, v in c.kind.value_counts().items()))
    print(f"cases: natural {(c[c.kind == 'case'].cause == 'Natural').sum()}, human {(c[c.kind == 'case'].cause == 'Human').sum()}")
    # sanity: the weather at ignition versus the same point's other hours
    cs = c[c.kind == "case"]; tc = c[c.kind == "t_ctrl"]
    print(f"median FFWI at ignition {cs.ffwi.median():.1f} vs same point other hours {tc.ffwi.median():.1f}; RH {cs.rh.median():.0f} vs {tc.rh.median():.0f}; precip72 {cs.precip72_mm.median():.1f} vs {tc.precip72_mm.median():.1f}")
    for cause in ["all", "Natural", "Human"]:
        d = c if cause == "all" else c[c.cause == cause]
        print(f"\n== {cause}: AUC, held out by HRRR square ==")
        print(f"{'features':8s} " + " ".join(f"{k:>12s}" for k in ["vs t_ctrl", "vs s_ctrl", "vs r_ctrl"]))
        for name, parts in SETS.items():
            cols = sum((FEATS[p] for p in parts), [])
            row = []
            for ctrl in ["t_ctrl", "s_ctrl", "r_ctrl"]:
                dd = d[d.kind.isin(["case", ctrl])]
                X = dd[cols].to_numpy(np.float64)
                X = np.nan_to_num(X)
                y = (dd.kind == "case").to_numpy(int)
                row.append(auc_cv(X, y, dd.sq.to_numpy()))
            print(f"{name:8s} " + " ".join(f"{v:12.3f}" for v in row))


if __name__ == "__main__":
    sys.exit(main())
