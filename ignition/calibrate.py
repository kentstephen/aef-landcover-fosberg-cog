# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "pyproj"]
# ///
"""Are the ignition head's probabilities real? The head (models/ignition_logit.json) was fit case
against control, one to one, so its 50% means "as fire-like as a typical control hour 20 to 30 km
away", not an hourly chance. This turns the logit into a rate and checks it:

  1. The population: cases_pop.parquet, random CONUS land pixel-hours 2019 to the last complete HRRR
     block (make_population.py), with weather, embedding and NLCD sampled like the cases. Each row
     stands for the same share of CONUS land x hours (self-weighting), so sum of lambda x weight over
     the rows is the model's expected count of ignitions.
  2. The offset c: lambda(z) = sigmoid(z + c) ignitions per km2 per hour, c solved so the expected
     count equals the WFIGS known-cause ignitions in the same window (what the head was fit on).
     Also c for all WFIGS wildfires (cause known or not).
  3. Checks, observed against expected: by decile of the population's logit (the reliability curve;
     the cases' logits are out of fold, a 5-fold refit by HRRR square), by region, by month, by
     NLCD class. Cases are weighted back to the WFIGS population (eligible fires in the case's
     weather group / cases drawn from it), since the groups were drawn by sqrt(fires) and capped.
     Poisson deviance of the model against the flat rate over region x month cells.
  4. What the numbers mean on the map: the rate per 1000 km2 per day at the logits the map draws.

Writes data/ignition/calibration.txt (this report) and models/ignition_calibration.json (the offset
and the per-class and per-region ratios). The head and the shader are unchanged.

    uv run ignition/calibrate.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit import CLASSES, D, FEATS, REGION_OF, load, load_set  # noqa: E402
from make_cases import CHUNK_T, CHUNK_XY, H, T0, W, hrrr_pixels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HEAD = ROOT / "models" / "ignition_logit.json"
OUT_TXT = D / "calibration.txt"
OUT_JSON = ROOT / "models" / "ignition_calibration.json"
NAMES = {11: "water", 12: "snow", 21: "dev open", 22: "dev low", 23: "dev med", 24: "dev high", 31: "barren", 41: "deciduous", 42: "evergreen",
         43: "mixed", 52: "shrub", 71: "grass", 81: "pasture", 82: "crops", 90: "woody wet", 95: "herb wet"}
lines = []


def say(s=""):
    print(s)
    lines.append(s)


def head_logit(d, m):
    cols = FEATS["P"] + m["weather"]
    X = np.nan_to_num(d[cols].to_numpy(np.float64))
    return ((X - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"][0]) + m["intercept"][0]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def wfigs_eligible(t_start, t_end):
    """WFIGS wildfires in the HRRR grid and the window, with the case filters (known cause) and without."""
    w = pd.read_parquet(D / "wfigs.parquet")
    w = w[w.IncidentTypeCategory == "WF"].copy()
    w["time"] = w.FireDiscoveryDateTime.dt.floor("h")
    w["ti"] = ((w.time - T0) / pd.Timedelta("1h")).astype(int)
    hx, hy = hrrr_pixels(w.geom_lon.values, w.geom_lat.values)
    w["hx"], w["hy"] = hx, hy
    w = w[(hx >= 0) & (hx < W) & (hy >= 0) & (hy < H) & (w.ti >= t_start) & (w.ti < t_end)].copy()
    w["group"] = (w.ti // CHUNK_T) * 100000 + (w.hy // CHUNK_XY) * 100 + (w.hx // CHUNK_XY)
    w["region"] = w.POOState.map(REGION_OF).fillna("Northeast")
    w["month"] = w.time.dt.month
    known = w[w.FireCause.isin(["Natural", "Human"])]
    return known, w


def table(name, keys, exp, obs, extra=None):
    say(f"\n-- {name}: observed / expected --")
    say(f"{'':12s} {'observed':>9s} {'expected':>9s} {'ratio':>7s}" + (f" {'per 1000 km2 day':>17s}" if extra is not None else ""))
    for k in keys:
        e, o = exp.get(k, 0.0), obs.get(k, 0.0)
        r = o / e if e > 0 else float("nan")
        row = f"{str(k):12s} {o:9.0f} {e:9.0f} {r:7.2f}"
        if extra is not None:
            row += f" {extra.get(k, float('nan')):17.4f}"
        say(row)


def main():
    m = json.loads(HEAD.read_text())
    pop = load_set("pop")
    if pop is None:
        raise SystemExit("the population needs weather_pop, embeddings_pop and nlcd_pop (the three samplers with --tag pop)")
    raw = pd.read_parquet(D / "cases_pop.parquet")
    land_km2, hours, t_start, t_end = float(raw.land_km2.iloc[0]), int(raw.hours.iloc[0]), int(raw.t_start.iloc[0]), int(raw.t_end.iloc[0])
    pop["region"] = pop.state.map(REGION_OF).fillna("Northeast")
    pop["month"] = pop.time.dt.month
    pop["z"] = head_logit(pop, m)
    n_pop = len(pop)
    w_pop = land_km2 * hours / n_pop  # km2 x hours per row
    pop["w"] = w_pop
    say(f"population: {len(raw)} pixel-hours drawn, {n_pop} with weather, embedding and NLCD; each stands for {w_pop:,.0f} km2 h "
        f"({land_km2:,.0f} km2 x {hours} h, {T0 + pd.Timedelta(hours=t_start):%Y-%m-%d} to {T0 + pd.Timedelta(hours=t_end):%Y-%m-%d})")
    say(f"head: {m['teacher']}; population logit p1 {np.percentile(pop.z, 1):.2f}, median {np.median(pop.z):.2f}, p99 {np.percentile(pop.z, 99):.2f}, max {pop.z.max():.2f}")

    # observed
    known, allwf = wfigs_eligible(t_start, t_end)
    n_known, n_all = len(known), len(allwf)
    say(f"WFIGS wildfires in the grid and window: {n_all} ({n_known} with a known cause: {int((known.FireCause == 'Natural').sum())} natural, "
        f"{int((known.FireCause == 'Human').sum())} human); CONUS-wide that is {n_known / (land_km2 / 1000) / (hours / 24):.4f} known-cause "
        f"ignitions per 1000 km2 per day, {n_all / (land_km2 / 1000) / (hours / 24):.4f} all")

    # the offset
    def expected(c):
        return float((sigmoid(pop.z.values + c) * w_pop).sum())
    c_known = brentq(lambda c: expected(c) - n_known, -40, 10)
    c_all = brentq(lambda c: expected(c) - n_all, -40, 10)
    say(f"\noffset c (lambda = sigmoid(z + c) ignitions per km2 per hour): {c_known:.3f} for known-cause ignitions, {c_all:.3f} for all WFIGS wildfires")
    say(f"  per 1000 km2 per day: multiply lambda by 24000 (add {np.log(24000):.2f} to the logit); per 10 m pixel-hour: lambda x 1e-4 (add {np.log(1e-4):.2f})")
    say(f"  so the map's 50% (z = 0) is {sigmoid(c_known) * 24000:.4f} known-cause ignitions per 1000 km2 per day; the population median logit "
        f"{np.median(pop.z):.2f} is {sigmoid(np.median(pop.z) + c_known) * 24000:.4f}; the top 1% ({np.percentile(pop.z, 99):.2f}) is {sigmoid(np.percentile(pop.z, 99) + c_known) * 24000:.4f}")
    pop["lam"] = sigmoid(pop.z.values + c_known)
    pop["exp"] = pop.lam * w_pop

    # the cases, out of fold, weighted back to WFIGS
    big = load("big")
    d = big[big.kind.isin(["case", "r30_ctrl"])].copy()
    y = (d.kind == "case").to_numpy(int)
    cols = FEATS["P"] + m["weather"]
    X = np.nan_to_num(d[cols].to_numpy(np.float64))
    z_oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, y, d.sq.to_numpy()):
        mm = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(X[tr], y[tr])
        z_oof[te] = mm.decision_function(X[te])
    d["z"] = z_oof
    cases = d[d.kind == "case"].copy()
    elig = known.groupby("group").size()
    drawn = cases.groupby("group").size()
    cases["wt"] = cases.group.map(elig / drawn).fillna(1.0)
    cases["month"] = cases.time.dt.month
    say(f"\ncases: {len(cases)} with out-of-fold logits, weights sum {cases.wt.sum():,.0f} (WFIGS known-cause {n_known}); "
        f"case logit median {cases.z.median():.2f} vs population {pop.z.median():.2f}; in-sample head on the cases median {head_logit(cases, m).mean():.2f}")

    # reliability by decile of the population logit
    say("\n-- reliability: bins of the population's logit; expected = model, observed = WFIGS ignitions (weighted cases) falling in the bin --")
    edges = np.quantile(pop.z, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, .999, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-95", "95-99", "99-99.9", "99.9-100"]
    pb = pd.cut(pop.z, edges, labels=labels, include_lowest=True)
    cb = pd.cut(cases.z, edges, labels=labels, include_lowest=True)
    exp_b = pop.groupby(pb, observed=False).exp.sum()
    obs_b = cases.groupby(cb, observed=False).wt.sum() * (n_known / cases.wt.sum())
    say(f"{'percentile':12s} {'logit from':>10s} {'expected':>9s} {'observed':>9s} {'ratio':>6s} {'share exp':>9s} {'share obs':>9s}")
    for i, lab in enumerate(labels):
        e, o = exp_b.get(lab, 0.0), obs_b.get(lab, 0.0)
        say(f"{lab:12s} {edges[i]:10.2f} {e:9.0f} {o:9.0f} {o / e if e else float('nan'):6.2f} {e / n_known:9.1%} {o / n_known:9.1%}")
    top10 = obs_b[["90-95", "95-99", "99-99.9", "99.9-100"]].sum() / n_known
    top1 = obs_b[["99-99.9", "99.9-100"]].sum() / n_known
    say(f"share of ignitions in the top 10% of pixel-hours {top10:.1%} (model expects {exp_b[['90-95', '95-99', '99-99.9', '99.9-100']].sum() / n_known:.1%}); "
        f"top 1% {top1:.1%} (model {exp_b[['99-99.9', '99.9-100']].sum() / n_known:.1%})")

    # offset AND slope: lambda = exp(a + b z) (Poisson likelihood: the weighted cases against the population), is the head too flat or too steep
    def fit_ab(b):
        a = np.log(n_known / (np.exp(b * pop.z.values) * w_pop).sum())
        ll = (cases.wt.values * (a + b * cases.z.values)).sum() * (n_known / cases.wt.sum()) - n_known
        return a, ll
    bs = np.linspace(0.5, 2.5, 201)
    lls = [fit_ab(b)[1] for b in bs]
    b_hat = float(bs[int(np.argmax(lls))]); a_hat = fit_ab(b_hat)[0]
    say(f"\noffset and slope (lambda = exp(a + b z)): b = {b_hat:.2f}, a = {a_hat:.3f} (b = 1 is the plain offset; b > 1 means the head is too flat, "
        f"its logit differences understate the real odds). Rescaled reliability:")
    pop["exp2"] = np.exp(a_hat + b_hat * pop.z.values) * w_pop
    exp2_b = pop.groupby(pb, observed=False).exp2.sum()
    say(f"{'percentile':12s} {'expected':>9s} {'observed':>9s} {'ratio':>6s}")
    for lab in labels:
        e, o = exp2_b.get(lab, 0.0), obs_b.get(lab, 0.0)
        say(f"{lab:12s} {e:9.0f} {o:9.0f} {o / e if e else float('nan'):6.2f}")

    # by region, by month, by class
    regions = ["Pacific", "Mountain", "Southwest", "Plains", "Southeast", "Northeast"]
    exp_r = pop.groupby("region").exp.sum()
    obs_r = known.groupby("region").size().astype(float)
    area_r = pop.groupby("region").w.sum() / hours
    rate_r = {r: obs_r.get(r, 0) / (area_r.get(r, np.nan) / 1000) / (hours / 24) for r in regions}
    table("region", regions, exp_r, obs_r, rate_r)
    say(f"  (all-cause WFIGS by region: " + ", ".join(f"{r} {int((allwf.region == r).sum())}" for r in regions) + ")")
    exp_m = pop.groupby("month").exp.sum()
    obs_m = known.groupby("month").size().astype(float)
    table("month", list(range(1, 13)), exp_m, obs_m)
    exp_c = pop.groupby("nlcd").exp.sum()
    obs_c = cases.groupby("nlcd").wt.sum() * (n_known / cases.wt.sum())
    area_c = pop.groupby("nlcd").w.sum() / hours
    rate_c = {c: obs_c.get(c, 0) / (area_c.get(c, np.nan) / 1000) / (hours / 24) for c in CLASSES}
    say("\n(classes: the NLCD pixel at the point; observed from the weighted cases, so a class's observed count has the cases' sampling noise)")
    table("NLCD class", [NAMES[c] for c in CLASSES if c in exp_c.index or c in obs_c.index], exp_c.rename(index=NAMES), obs_c.rename(index=NAMES), {NAMES[k]: v for k, v in rate_c.items()})
    land = pd.read_parquet(D / "land_area.parquet")
    land["region"] = land.state.map(REGION_OF).fillna("Northeast")
    say("\n-- land area per class, NLCD 2024 (km2), the population's share of it vs the map's: --")
    la = land.groupby("nlcd").km2.sum()
    ps = pop.groupby("nlcd").size() / n_pop
    say(f"{'class':12s} {'km2':>12s} {'map share':>10s} {'pop share':>10s}")
    for c in CLASSES:
        say(f"{NAMES[c]:12s} {la.get(c, 0):12,.0f} {la.get(c, 0) / la.sum():10.1%} {ps.get(c, 0):10.1%}")

    # deviance against the flat rate over region x month cells
    cell_e = pop.groupby(["region", "month"]).exp.sum()
    cell_o = known.groupby(["region", "month"]).size().astype(float)
    cell_n = pop.groupby(["region", "month"]).w.sum() * (n_known / (land_km2 * hours))  # flat rate
    idx = cell_e.index.union(cell_o.index).union(cell_n.index)
    o = cell_o.reindex(idx, fill_value=0.0); e = cell_e.reindex(idx, fill_value=1e-9); f = cell_n.reindex(idx, fill_value=1e-9)

    def dev(o, mu):
        mu = np.maximum(mu, 1e-9)
        return float(2 * np.sum(np.where(o > 0, o * np.log(o / mu), 0) - (o - mu)))
    say(f"\nPoisson deviance over {len(idx)} region x month cells: model {dev(o, e):,.0f}, flat rate {dev(o, f):,.0f}, saturated 0; "
        f"the model explains {1 - dev(o, e) / dev(o, f):.1%} of the flat rate's deviance")

    # what it means on the map
    say("\n-- on the map: the rate at the logits it draws (known-cause offset) --")
    say(f"{'logit':>6s} {'head %':>7s} {'per 1000 km2 day':>17s} {'per 10 m px hour':>17s} {'x the CONUS mean':>17s}")
    mean_rate = n_known / (land_km2 / 1000) / (hours / 24)
    for z in [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
        lam = sigmoid(z + c_known)
        say(f"{z:6d} {sigmoid(z):7.1%} {lam * 24000:17.4f} {lam * 1e-4:17.2e} {lam * 24000 / mean_rate:17.1f}")

    out = {
        "head": m["teacher"], "unit": "ignitions per km2 per hour = sigmoid(head logit + offset)",
        "offset_known_cause": c_known, "offset_all_wfigs": c_all, "slope_fit": {"a": float(a_hat), "b": b_hat, "unit": "lambda = exp(a + b z) per km2 per hour"},
        "to_per_1000km2_day": float(np.log(24000)), "to_per_10m_pixel_hour": float(np.log(1e-4)),
        "n_known": n_known, "n_all": n_all, "land_km2": land_km2, "hours": hours, "population_rows": n_pop,
        "conus_mean_known_per_1000km2_day": mean_rate,
        "ratio_by_region": {r: float(obs_r.get(r, 0) / exp_r.get(r, np.nan)) for r in regions},
        "ratio_by_month": {int(k): float(obs_m.get(k, 0) / exp_m.get(k, np.nan)) for k in range(1, 13)},
        "ratio_by_class": {int(c): float(obs_c.get(c, 0) / exp_c.get(c, np.nan)) for c in CLASSES if c in exp_c.index},
        "share_in_top10pct": float(top10), "share_in_top1pct": float(top1),
        "deviance_model": dev(o, e), "deviance_flat": dev(o, f),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    OUT_TXT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {OUT_TXT} and {OUT_JSON}")


if __name__ == "__main__":
    sys.exit(main())
