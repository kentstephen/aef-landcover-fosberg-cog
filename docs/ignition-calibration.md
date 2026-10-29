# Are the ignition head's probabilities real? (calibration check, 2026-10-29)

The map's ignition head (`models/ignition_logit.json`) was fit case against control, one to one: 22,204
WFIGS ignitions (2019 to 2026, known cause) against as many control pixel-hours 20 to 30 km away at
another hour of the same 90-day block. So its 50% means "as fire-like as a typical nearby control
hour", not an hourly chance, and the regional base rate cancels out (the controls are local). This
check turns the logit into a rate of ignitions per area and time and compares it with what WFIGS
recorded. The head and the shader are unchanged; the numbers below are what the map would read if
its scale were calibrated.

## Method

1. A population sample of real pixel-hours (`ignition/make_population.py`): 19,988 random points on
   CONUS land at random hours from 2019-01-01 to 2026-10-28, stratified by HRRR weather group (a
   135 km square x 90 days, 400 groups x 50), self-weighting, so each row stands for 26.9 million
   km2 h. Weather, embedding (of the year before) and NLCD sampled with the same three samplers as
   the cases (`--tag pop`).
2. The offset: lambda(z) = sigmoid(z + c) ignitions per km2 per hour, c solved so the expected count
   over the population equals the WFIGS known-cause ignitions in the same grid and window. Also an
   offset-plus-slope fit (lambda = exp(a + b z), Poisson likelihood) to see whether the head is too
   flat or too steep.
3. Observed against expected: by percentile of the population logit (the cases' logits are out of
   fold, a 5-fold refit by HRRR square; cases weighted back to the WFIGS population by eligible fires
   in their group over cases drawn), by region, by month, by NLCD class. Poisson deviance of the
   model against a flat rate over region x month cells.
4. Land area per class from the NLCD 2024 mosaic at its 480 m overview (`ignition/land_area.py`),
   states rasterized from the Census boundaries: 7.84 million km2, matching the state polygons.

`ignition/calibrate.py` writes the full report to `data/ignition/calibration.txt` and the offsets and
ratios to `models/ignition_calibration.json`.

## Numbers

WFIGS wildfires in the HRRR grid and window: 287,677, of which 152,333 with a known cause (42,072
natural, 110,261 human). CONUS-wide that is 0.0068 known-cause ignitions per 1000 km2 per day
(0.0128 all).

Offset c = -14.84 (known cause), -14.20 (all WFIGS). Per 1000 km2 per day add 10.09 to the logit;
per 10 m pixel-hour add -9.21. The head's 50% (logit 0) is 0.0086 known-cause ignitions per 1000 km2
per day, 1.3 times the CONUS mean. The population median logit (-1.30) is 0.0024; the top 1%
(logit 2.14) is 0.073; logit 4 is 0.47 (69 times the mean); logit 6 is 3.5 (511 times).

Offset and slope: b = 1.06, a = -14.89. The slope is essentially 1, so a linear rescaling of the logit
does not fix what is wrong below.

### Reliability by percentile of the population logit (known-cause offset)

| percentile | logit from | expected | observed | ratio |
|---|---|---|---|---|
| 0-10 | | 595 | 131 | 0.22 |
| 10-20 | -2.98 | 1364 | 411 | 0.30 |
| 20-30 | -2.40 | 2184 | 732 | 0.33 |
| 30-40 | -1.99 | 3176 | 1561 | 0.49 |
| 40-50 | -1.63 | 4486 | 2415 | 0.54 |
| 50-60 | -1.30 | 6333 | 4546 | 0.72 |
| 60-70 | -0.94 | 9231 | 8185 | 0.89 |
| 70-80 | -0.55 | 14033 | 13657 | 0.97 |
| 80-90 | -0.10 | 23495 | 25661 | 1.09 |
| 90-95 | 0.51 | 20962 | 24622 | 1.17 |
| 95-99 | 1.07 | 35207 | 41167 | 1.17 |
| 99-99.9 | 2.14 | 23823 | 23001 | 0.97 |
| 99.9-100 | 3.19 | 7445 | 6246 | 0.84 |

The top 10% of pixel-hours hold 62% of the ignitions (the model expects 57%); the top 1% hold 19%
(model 21%). The ordering is good. The shape is a hump: the bottom half is overpredicted 2 to 5
times, the 80th to 99th percentiles underpredicted by about 15%, the extreme top overpredicted.

### By region (observed / expected, and the observed rate)

| region | observed | expected | ratio | per 1000 km2 day |
|---|---|---|---|---|
| Pacific | 27589 | 15471 | 1.78 | 0.0112 |
| Mountain | 46308 | 18923 | 2.45 | 0.0101 |
| Southwest | 33623 | 32854 | 1.02 | 0.0079 |
| Plains | 19730 | 33514 | 0.59 | 0.0041 |
| Southeast | 13164 | 31390 | 0.42 | 0.0035 |
| Northeast | 11919 | 20182 | 0.59 | 0.0049 |

### By month (observed / expected)

Jan 0.63, Feb 0.84, Mar 1.02, Apr 1.01, May 0.98, Jun 0.88, Jul 1.30, Aug 1.49, Sep 0.86, Oct 0.69,
Nov 0.70, Dec 0.46.

### By NLCD class (observed / expected; observed from the weighted cases)

| class | observed | expected | ratio | per 1000 km2 day |
|---|---|---|---|---|
| dev open | 16436 | 15022 | 1.09 | 0.0238 |
| dev low | 17558 | 22428 | 0.78 | 0.0348 |
| dev med | 3516 | 12586 | 0.28 | 0.0140 |
| dev high | 255 | 1055 | 0.24 | 0.0043 |
| barren | 812 | 746 | 1.09 | 0.0035 |
| deciduous | 8133 | 10042 | 0.81 | 0.0038 |
| evergreen | 38679 | 13389 | 2.89 | 0.0161 |
| mixed | 3024 | 3225 | 0.94 | 0.0046 |
| shrub | 32544 | 23955 | 1.36 | 0.0067 |
| grass | 14765 | 18337 | 0.81 | 0.0045 |
| pasture | 7148 | 11367 | 0.63 | 0.0047 |
| crops | 3299 | 12993 | 0.25 | 0.0008 |
| woody wet | 3862 | 4198 | 0.92 | 0.0040 |
| herb wet | 1845 | 2268 | 0.81 | 0.0048 |
| water | 448 | 708 | 0.63 | 0.0009 |

Poisson deviance over 72 region x month cells: model 100,246 against a flat rate's 136,762; the
model explains 27% of the flat rate's deviance at that grain.

## Reading

- The head ranks pixel-hours well, as the AUCs said. As a rate it is wrong in a way one offset cannot
  fix: it understates the West (Mountain 2.4 times, Pacific 1.8) and evergreen forest (2.9 times),
  and overstates cropland, medium and high density built land (4 times), the Plains, the Southeast
  and winter. Local controls cancel the regional base rate, so the head knows "fire-like for here",
  not "how much fire here gets". The hump in the reliability curve is the same thing seen along the
  logit axis.
- The per-class offsets the plan allowed for (one more row in the lookup texture, like the hold
  hours) would correct the class column directly. A per-region offset needs a region per pixel in the
  shader (a coarse texture, or the HRRR node), which the weather texture could carry.
- WFIGS under-reports outside federal land, badly in the Southeast (Georgia has 610 fires in 8
  years in WFIGS). The Southeast and Plains ratios are partly that. The Mountain and Pacific
  excess is real: those are where WFIGS is most complete and the head still comes out low.
- July and August are underpredicted 1.3 to 1.5 times with the weather in the head. Either the
  weather term is too weak at the extremes (the boosted tree beat the linear head on weather alone,
  natural fires 0.78 vs 0.71 AUC), or summer lightning is the missing piece: the cases are 72% human.
- On the map the calibrated numbers are tiny per pixel-hour (1e-11 to 1e-9). The readable unit is
  ignitions per 1000 km2 per day, or the multiple of the CONUS mean (logit 0 is 1.3 times the mean,
  logit 2 is 9, logit 4 is 69). The gate and the logit scale are unchanged by any of this.
