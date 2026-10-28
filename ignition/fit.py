# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn"]
# ///
"""Does the ground, read from the embeddings, say where fires start? Tiny heads on the pilot cases,
each case against each kind of control:

  case vs t_ctrl   same point, another hour: only the weather differs (the ground cannot help; AUC 0.5
                   for ground features is the check that nothing leaks)
  case vs s_ctrl   same hour, a point up to 2.5 km away: only the ground differs (the weather cannot help)
  case vs r_ctrl   another point and another hour: both differ
  s10/r10, s30/r30 the same at 5 to 10 km and 20 to 30 km (cases_wide.parquet, when sampled)

Feature sets: W weather (FFWI sustained and gust, prior-24 h max, 72 h precipitation, T, RH, wind),
E embedding (the 150 m unit mean, 64 bands, standardized), P the single 10 m pixel, N the NLCD
baseline (pixel class one-hot + 5 x 5 class fractions), ExW the products of the embedding with FFWI
(does the ground change how much the weather matters), D the change of the embedding over the two
years before the fire (when embeddings_b1.parquet exists), and sums. Logistic heads, plus one boosted
tree head (GB) on E+W. Held out by HRRR square (GroupKFold, 5 folds), so nearby fires never sit on
both sides. Reports AUC per cause.

    uv run ignition/fit.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parent.parent / "data" / "ignition"
CLASSES = [11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
GROUND = {"case": "a", "t_ctrl": "a", "s_ctrl": "b", "r_ctrl": "b"}


def dequant(q):
    f = q.astype(np.float32) / 127.5
    return np.sign(f) * f * f


def gkey(kind):
    # the ground belongs to the point: t_ctrl shares the case's, each r ring shares its s ring's
    return GROUND.get(kind, kind[1:].replace("_ctrl", ""))


def unpack(e, prefix="e"):
    E = np.stack([np.frombuffer(b, np.float32) for b in e.win_mean])
    return pd.DataFrame(E, index=e.index, columns=[f"{prefix}{i}" for i in range(64)])


def load_set(tag=""):
    sfx = f"_{tag}" if tag else ""
    if not (D / f"nlcd{sfx}.parquet").exists() or not (D / f"embeddings{sfx}.parquet").exists() or not (D / f"weather{sfx}.parquet").exists():
        return None
    c = pd.read_parquet(D / f"cases{sfx}.parquet")
    w = pd.read_parquet(D / f"weather{sfx}.parquet")
    e = pd.read_parquet(D / f"embeddings{sfx}.parquet")
    n = pd.read_parquet(D / f"nlcd{sfx}.parquet")
    c = c.join(w, how="inner")
    key = c.case_id.astype(str) + "_" + c.kind.map(gkey)
    emb = unpack(e)
    P = dequant(np.stack([np.frombuffer(b, np.int8) for b in e.pix]))
    emb = emb.join(pd.DataFrame(P, index=e.index, columns=[f"p{i}" for i in range(64)]))
    emb["win_norm"] = e.win_norm.values
    emb["nodata"] = e.nodata.values
    for back in (1, 2):
        f = D / f"embeddings{sfx}_b{back}.parquet"
        if f.exists():
            eb = unpack(pd.read_parquet(f), prefix=f"b{back}_")
            emb = emb.join(eb, how="left")
    ground = emb.join(n, how="inner")
    ground["pkey"] = key.loc[ground.index]
    c["pkey"] = key
    c = c.merge(ground, on="pkey", how="inner", suffixes=("", "_g"))
    c = c[~c.nodata & c.ffwi.notna() & (c.nlcd != 250)]
    return c


def load():
    c = load_set()
    wide = load_set("wide")
    if wide is not None:
        print(f"wide controls: {len(wide)} rows")
        c = pd.concat([c, wide], ignore_index=True)
    for cl in CLASSES:
        c[f"nlcd_is{cl}"] = (c.nlcd == cl).astype(float)
    c["sq"] = c.cy * 100 + c.cx
    # interactions: each embedding band times FFWI (standardized over all rows)
    fz = (c.ffwi - c.ffwi.mean()) / c.ffwi.std()
    for i in range(64):
        c[f"x{i}"] = c[f"e{i}"] * fz
    # the trajectory: the change from two years before to the year before, and the cosine between them
    if "b1_0" in c:
        E1 = c[[f"e{i}" for i in range(64)]].to_numpy()
        E2 = c[[f"b1_{i}" for i in range(64)]].to_numpy()
        for i in range(64):
            c[f"d{i}"] = E1[:, i] - E2[:, i]
        c["cos12"] = (E1 * E2).sum(1)
        if "b2_0" in c:
            E3 = c[[f"b2_{i}" for i in range(64)]].to_numpy()
            c["cos23"] = (E2 * E3).sum(1)
            c["cos13"] = (E1 * E3).sum(1)
    return c


FEATS = {
    "W": ["ffwi", "ffwi_gust", "ffwi_max24", "precip72_mm", "t_c", "rh", "wind_ms"],
    "E": [f"e{i}" for i in range(64)] + ["win_norm"],
    "P": [f"p{i}" for i in range(64)],
    "N": [f"nlcd_is{cl}" for cl in CLASSES] + [f"nlcd_f{cl}" for cl in CLASSES],
    "ExW": [f"x{i}" for i in range(64)],
    "D": [f"d{i}" for i in range(64)] + ["cos12"],
    "C": ["cos12", "cos23", "cos13"],
}
SETS = {"W": ["W"], "N": ["N"], "E": ["E"], "P": ["P"], "E+P": ["E", "P"], "N+W": ["N", "W"], "E+W": ["E", "W"],
        "E+N": ["E", "N"], "E+N+W": ["E", "N", "W"], "E+W+ExW": ["E", "W", "ExW"], "GB E+W": ["E", "W"]}
TRAJ = {"D": ["D"], "C": ["C"], "E+D": ["E", "D"], "E+C": ["E", "C"], "E+W": ["E", "W"], "E+D+W": ["E", "D", "W"], "E+C+W": ["E", "C", "W"]}


def auc_cv(X, y, groups, gb=False):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if gb:
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, pred)


def table(d, sets, ctrls, title):
    ctrls = [k for k in ctrls if (d.kind == k).any()]
    print(f"\n== {title}: AUC, held out by HRRR square ==")
    print(f"{'features':9s} " + " ".join(f"{'vs ' + k.replace('_ctrl', ''):>9s}" for k in ctrls))
    for name, parts in sets.items():
        cols = sum((FEATS[p] for p in parts), [])
        if any(col not in d for col in cols):
            continue
        row = []
        for ctrl in ctrls:
            dd = d[d.kind.isin(["case", ctrl])].dropna(subset=cols)
            X = np.nan_to_num(dd[cols].to_numpy(np.float64))
            y = (dd.kind == "case").to_numpy(int)
            row.append(auc_cv(X, y, dd.sq.to_numpy(), gb=name.startswith("GB")))
        print(f"{name:9s} " + " ".join(f"{v:9.3f}" for v in row))


def main():
    c = load()
    print(f"rows {len(c)}: " + ", ".join(f"{k} {v}" for k, v in c.kind.value_counts().items()))
    print(f"cases: natural {(c[c.kind == 'case'].cause == 'Natural').sum()}, human {(c[c.kind == 'case'].cause == 'Human').sum()}")
    cs = c[c.kind == "case"]; tc = c[c.kind == "t_ctrl"]
    print(f"median FFWI at ignition {cs.ffwi.median():.1f} vs same point other hours {tc.ffwi.median():.1f}; RH {cs.rh.median():.0f} vs {tc.rh.median():.0f}; precip72 {cs.precip72_mm.median():.1f} vs {tc.precip72_mm.median():.1f}")
    ctrls = ["t_ctrl", "s_ctrl", "r_ctrl", "s10_ctrl", "r10_ctrl", "s30_ctrl", "r30_ctrl"]
    for cause in ["all", "Natural", "Human"]:
        d = c if cause == "all" else c[c.cause == cause]
        table(d, SETS, ctrls, cause)
    if "d0" in c:
        print(f"\ntrajectory rows: {c.d0.notna().sum()} with the year before that, {c.get('cos23', pd.Series(dtype=float)).notna().sum()} with two")
        for cause in ["all", "Natural", "Human"]:
            d = c if cause == "all" else c[c.cause == cause]
            table(d, TRAJ, ctrls, f"{cause}, trajectory")


if __name__ == "__main__":
    sys.exit(main())
