# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scikit-learn", "pillow"]
# ///
"""Fit tiny land cover heads on AEF embeddings and draw the land cover they read.

Reads data/tiles/*.npz from sample_landcover.py. Features: the 64-band embedding, dequantized and
unit-normalized (so the same head applies to 10 m pixels and to overview centroids alike).

Heads (all linear, so each is a few vec4 uniforms in a shader):
  nlcd  : multinomial logistic regression, 64 -> NLCD classes, taught by Annual NLCD 2024
  esri  : multinomial logistic regression, 64 -> Esri / IO classes, taught by IO LULC 2024
  canopy: ridge regression, 64 -> [cover fraction, mean height m, p95 height m], taught by Meta canopy v1

Training cells are the "pure" ones: at least PURE of the 16 children (10 m) carry the majority class.
Holdout tiles are never fit on; their accuracy is the honest number.

Writes models/*.json and, per tile, data/previews/<tile>_<head>.png: teacher | head, side by side.

    uv run scripts/fit_landcover_head.py [--holdout kansas_prairie,sierra_foothills] [--rows 1500000]

Peak memory: ROWS x 64 float32 (1.5 M rows = 384 MB) plus sklearn's copies, under 2 GB; previews
run one tile at a time (a 2048 x 2048 x 64 tile dequantized in 256-row strips).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
TILE_DIR = ROOT / "data" / "tiles"
MODEL_DIR = ROOT / "models"
PREVIEW_DIR = ROOT / "data" / "previews"
PURE = 12  # of 16 children

# Protan-safe palettes: no red, no red-vs-green pair. Greens carry the trees, purples the built
# classes, tans and yellows the open fuels, blues and teals the water and wetlands.
NLCD = {
    11: ("Open water", "#3b6fd1"), 12: ("Perennial ice/snow", "#ffffff"),
    21: ("Developed, open space", "#cdbddc"), 22: ("Developed, low intensity", "#a98bc4"),
    23: ("Developed, medium intensity", "#845aa8"), 24: ("Developed, high intensity", "#5a2d82"),
    31: ("Barren land", "#bdb6a4"),
    41: ("Deciduous forest", "#86cf86"), 42: ("Evergreen forest", "#1f6e3a"), 43: ("Mixed forest", "#4da35a"),
    52: ("Shrub/scrub", "#d8b365"), 71: ("Grassland/herbaceous", "#f2e394"),
    81: ("Pasture/hay", "#e8c34a"), 82: ("Cultivated crops", "#b8762a"),
    90: ("Woody wetlands", "#4fb3b3"), 95: ("Emergent herbaceous wetlands", "#a6dddd"),
}
ESRI = {
    1: ("Water", "#3b6fd1"), 2: ("Trees", "#1f6e3a"), 4: ("Flooded vegetation", "#4fb3b3"),
    5: ("Crops", "#b8762a"), 7: ("Built area", "#845aa8"), 8: ("Bare ground", "#bdb6a4"),
    9: ("Snow/ice", "#ffffff"), 10: ("Clouds", "#777777"), 11: ("Rangeland", "#f2e394"),
}
TEACHERS = {"nlcd": NLCD, "esri": ESRI}
CANOPY_TARGETS = ["cover", "avg_m", "p95_m"]


def hex_rgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def dequant(q):
    q = q.astype(np.float32)
    return np.sign(q) * (q / 127.5) ** 2


def features(emb_int8):
    v = dequant(emb_int8)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-6)


def majority(blocks):
    """blocks: (N, 16) uint8 labels -> (label, count) per row; label 0 where the block is all nodata."""
    n, k = blocks.shape
    lab = np.zeros(n, np.int64)
    cnt = np.zeros(n, np.int64)
    classes = np.unique(blocks)
    for c in classes:
        if c == 0:
            continue
        m = (blocks == c).sum(axis=1)
        better = m > cnt
        lab[better] = c
        cnt[better] = m[better]
    return lab, cnt


def load_samples(names):
    parts = []
    for name in names:
        d = np.load(TILE_DIR / f"{name}.npz")
        nl, nc = majority(d["s_nlcd"])
        el, ec = majority(d["s_esri"])
        parts.append(dict(
            name=name, emb=d["s_emb"], nlcd=nl, nlcd_pure=nc >= PURE, esri=el, esri_pure=ec >= PURE,
            canopy=np.stack([d["s_chm_cover"] / 1000.0, d["s_chm_avg"] / 100.0, d["s_chm_p95"] / 100.0], axis=1),
        ))
    return parts


def stack(parts, key):
    return np.concatenate([p[key] for p in parts])


def fit_classifier(parts, teacher, rows, rng, seed):
    emb = stack(parts, "emb")
    y = stack(parts, teacher)
    pure = stack(parts, f"{teacher}_pure")
    keep = pure & (y > 0) & np.isin(y, list(TEACHERS[teacher]))
    idx = np.flatnonzero(keep)
    if idx.size > rows:
        idx = rng.choice(idx, rows, replace=False)
    X = features(emb[idx])
    y = y[idx]
    classes, counts = np.unique(y, return_counts=True)
    print(f"  [{teacher}] fit rows {idx.size:,}; classes " + ", ".join(f"{c}:{n:,}" for c, n in zip(classes, counts)))
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    Xs = (X - mean) / scale
    t0 = time.time()
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", n_jobs=-1, random_state=seed)
    clf.fit(Xs, y)
    print(f"  [{teacher}] fit {time.time() - t0:.0f} s, train acc {clf.score(Xs, y):.3f}")
    return dict(kind="logit", teacher=teacher, mean=mean, scale=scale, coef=clf.coef_, intercept=clf.intercept_, classes=clf.classes_)


def predict_logit(m, X, chunk=262_144):
    out = np.empty(len(X), np.int64)
    for i in range(0, len(X), chunk):
        z = ((X[i:i + chunk] - m["mean"]) / m["scale"]) @ m["coef"].T + m["intercept"]
        out[i:i + chunk] = m["classes"][z.argmax(axis=1)]
    return out


def evaluate_classifier(m, parts, label):
    teacher = m["teacher"]
    for p in parts:
        keep = p[f"{teacher}_pure"] & np.isin(p[teacher], list(TEACHERS[teacher]))
        if keep.sum() == 0:
            continue
        X = features(p["emb"][keep])
        y = p[teacher][keep]
        pred = predict_logit(m, X)
        acc = (pred == y).mean()
        # per-class recall on the classes present
        cls = np.unique(y)
        rec = {int(c): round(float((pred[y == c] == c).mean()), 2) for c in cls if (y == c).sum() >= 200}
        print(f"  [{teacher}] {label} {p['name']}: acc {acc:.3f} on {keep.sum():,} pure cells; recall {rec}")


def fit_canopy(parts, rows, rng):
    emb = stack(parts, "emb")
    y = stack(parts, "canopy")
    keep = np.isfinite(y).all(axis=1)
    idx = np.flatnonzero(keep)
    if idx.size > rows:
        idx = rng.choice(idx, rows, replace=False)
    X = features(emb[idx])
    y = y[idx]
    mean = X.mean(axis=0)
    scale = X.std(axis=0) + 1e-6
    reg = Ridge(alpha=1.0).fit((X - mean) / scale, y)
    print(f"  [canopy] fit rows {idx.size:,}, train R2 {reg.score((X - mean) / scale, y):.3f}")
    return dict(kind="ridge", teacher="canopy", mean=mean, scale=scale, coef=reg.coef_, intercept=reg.intercept_, targets=CANOPY_TARGETS)


def evaluate_canopy(m, parts, label):
    for p in parts:
        y = p["canopy"]
        keep = np.isfinite(y).all(axis=1)
        X = features(p["emb"][keep])
        pred = ((X - m["mean"]) / m["scale"]) @ m["coef"].T + m["intercept"]
        yk = y[keep]
        r2 = 1 - ((pred - yk) ** 2).sum(axis=0) / ((yk - yk.mean(axis=0)) ** 2).sum(axis=0)
        mae = np.abs(pred - yk).mean(axis=0)
        print(f"  [canopy] {label} {p['name']}: R2 cover {r2[0]:.2f} avg {r2[1]:.2f} p95 {r2[2]:.2f}; MAE cover {mae[0]:.2f} avg {mae[1]:.1f} m p95 {mae[2]:.1f} m")


def export(m, path):
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}
    out["inputs"] = "AEF 64 bands int8 -> sign(q)*(q/127.5)^2 -> unit length -> (x - mean) / scale"
    if m["kind"] == "logit":
        out["p"] = "softmax(coef . x + intercept) over classes"
        out["legend"] = {str(c): {"name": TEACHERS[m["teacher"]][int(c)][0], "color": TEACHERS[m["teacher"]][int(c)][1]} for c in m["classes"]}
    else:
        out["p"] = "coef . x + intercept, per target (cover 0..1, heights in m)"
    json.dump(out, open(path, "w"))
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1e3:.0f} KB)")


def teacher_mode_40m(a10, classes):
    """Mode of each 4x4 block of 10 m labels, 0 where empty."""
    side = a10.shape[0] // 4
    b = a10.reshape(side, 4, side, 4)
    lab = np.zeros((side, side), np.uint8)
    cnt = np.zeros((side, side), np.uint8)
    for c in classes:
        m = (b == c).sum(axis=(1, 3)).astype(np.uint8)
        better = m > cnt
        lab[better] = c
        cnt[better] = m[better]
    return lab


def paint(lab, legend):
    lut = np.zeros((256, 3), np.uint8)
    for c, (_, h) in legend.items():
        lut[c] = hex_rgb(h)
    return lut[lab]


def preview(name, models):
    d = np.load(TILE_DIR / f"{name}.npz")
    aef = d["aef40"]
    valid = d["valid"]
    side = aef.shape[1]
    for key, m in models.items():
        if m["kind"] != "logit":
            continue
        pred = np.zeros((side, side), np.uint8)
        for r0 in range(0, side, 256):  # strips keep the float copy small
            X = features(np.ascontiguousarray(aef[:, r0:r0 + 256, :]).transpose(1, 2, 0).reshape(-1, 64))
            pred[r0:r0 + 256] = predict_logit(m, X).reshape(-1, side)
        pred[~valid] = 0
        legend = TEACHERS[m["teacher"]]
        truth = teacher_mode_40m(d[f"{m['teacher']}10"], list(legend))
        agree = (pred == truth)[valid & (truth > 0)].mean()
        img = np.concatenate([paint(truth, legend), np.full((side, 8, 3), 20, np.uint8), paint(pred, legend)], axis=1)
        Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.NEAREST).save(PREVIEW_DIR / f"{name}_{key}.png")
        print(f"  preview {name}_{key}.png: teacher | head, 40 m, agreement with teacher mode {agree:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="kansas_prairie,sierra_foothills")
    ap.add_argument("--rows", type=int, default=1_500_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-previews", action="store_true")
    a = ap.parse_args()
    MODEL_DIR.mkdir(exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    names = sorted(p.stem for p in TILE_DIR.glob("*.npz"))
    hold = [h for h in a.holdout.split(",") if h in names]
    train = [n for n in names if n not in hold]
    print(f"tiles: train {train}; holdout {hold}")
    parts = load_samples(names)
    tr = [p for p in parts if p["name"] in train]
    ho = [p for p in parts if p["name"] in hold]

    models = {}
    for teacher in ("nlcd", "esri"):
        m = fit_classifier(tr, teacher, a.rows, rng, a.seed)
        evaluate_classifier(m, ho, "HOLDOUT")
        evaluate_classifier(m, tr[:3], "train-tile")
        export(m, MODEL_DIR / f"landcover_{teacher}_logit.json")
        models[teacher] = m
    m = fit_canopy(tr, a.rows, rng)
    evaluate_canopy(m, ho, "HOLDOUT")
    evaluate_canopy(m, tr[:3], "train-tile")
    export(m, MODEL_DIR / "canopy_ridge.json")
    models["canopy"] = m

    if not a.no_previews:
        for name in names:
            preview(name, models)


if __name__ == "__main__":
    main()
