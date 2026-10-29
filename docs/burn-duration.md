# Burn duration and size from the embedding

The ignition head says where a fire could start. This is the other half: how long and how big a fire
that starts here gets, read from the same 10 m AlphaEarth embedding, and what that gives the map.
Script: `ignition/burn_head.py`. Report: `data/ignition/burn.txt`. Model file: `models/burn_duration.json`.

## Data

The 22,204 ignition cases of the big set (WFIGS wildfires 2019 to 2026 with their 10 m pixel embedding,
the 150 m mean, the prior-year NLCD class at the point and the HRRR analysis weather at the hour), joined
back to WFIGS by point, name and year for the fire's out time. 17,701 have an out time after discovery
and within a year; 21,460 have an IncidentSize. Duration is discovery to out in hours. The out time is
the reporting agency's, so it runs past the flames, and it is the same clock for every class.

Duration over all fires: a quarter are out within 19 hours, half within 74, three quarters within 240.
Seven in ten are still open after a day.

## By land cover at the ignition point

| class | fires | p25 h | median h | p75 h | still open after 24 h | after 7 d | median acres | 100 acres or more |
|---|---|---|---|---|---|---|---|---|
| developed open | 1779 | 3.2 | 50.5 | 187 | 0.62 | 0.26 | 0.25 | 0.045 |
| developed low | 1938 | 2.1 | 45.1 | 169 | 0.59 | 0.25 | 0.20 | 0.028 |
| developed medium | 348 | 1.5 | 27.0 | 120 | 0.54 | 0.19 | 0.10 | 0.026 |
| developed high | 20 | 1.0 | 3.2 | 100 | 0.30 | 0.20 | 0.23 | 0.000 |
| barren | 117 | 2.2 | 62.5 | 138 | 0.60 | 0.23 | 0.10 | 0.056 |
| deciduous forest | 866 | 24.0 | 93.8 | 263 | 0.75 | 0.36 | 1.20 | 0.116 |
| evergreen forest | 4412 | 27.3 | 99.5 | 291 | 0.78 | 0.37 | 0.10 | 0.049 |
| mixed forest | 329 | 1.7 | 26.6 | 148 | 0.53 | 0.22 | 0.30 | 0.053 |
| shrub | 4373 | 23.2 | 95.5 | 253 | 0.74 | 0.36 | 0.25 | 0.080 |
| grassland | 1791 | 20.2 | 90.1 | 289 | 0.72 | 0.37 | 1.77 | 0.144 |
| pasture | 707 | 4.0 | 45.2 | 141 | 0.60 | 0.21 | 2.00 | 0.041 |
| crops | 325 | 3.3 | 45.4 | 160 | 0.59 | 0.24 | 1.00 | 0.089 |
| woody wetland | 396 | 5.8 | 51.2 | 234 | 0.65 | 0.31 | 0.50 | 0.064 |
| herbaceous wetland | 226 | 20.4 | 78.5 | 280 | 0.70 | 0.35 | 1.50 | 0.150 |
| open water | 74 | 1.6 | 24.6 | 137 | 0.53 | 0.24 | 0.10 | 0.025 |

Timber, shrub and grass fires run three to four days by the agency's clock and a quarter of them are
out within a day; built-ground, pasture and crop fires run about two days and a quarter are out within
four hours. Grassland and the herbaceous wetlands grow largest (one in seven reaches 100 acres);
evergreen fires are many and mostly tiny (median a tenth of an acre) but a twentieth reach 100 acres.

## The hold hours

The map's afterglow holds a past hour's weather term by the land cover: the ground stays lit for the
class's hold hours after its peak, sliding back under the gate over that time. The hold hours in
`web/src/aef/fuel.ts` are now the class medians above, capped at four days (the lookup texture holds a
byte of half hours): developed open 51, low 45, medium 27, high 3; barren 63; deciduous 94, evergreen 96,
mixed 27; shrub 95; grassland 90; pasture 45; crops 45; woody wetland 51; herbaceous wetland 78; water
and ice 0. The Esri classes take their NLCD counterparts. The first-cut table (grass 3 h, shrub 6,
evergreen 24, built 0) was shorter on every wildland class; with the default 24 h lookback the wildland
ground now keeps the day's peak through the day.

## Heads: does the embedding say more about burn than the class

Ridge regression to log duration and log10 size, and logistic regression to "still open after 24 h" and
"100 acres or more", five folds held out by HRRR square (no square in both the fit and the test).
N = the NLCD class (one-hot and the 5 x 5 fractions), W = the hour's FFWI, temperature, humidity and
wind, P = the 10 m pixel embedding, E = the 150 m mean embedding.

| features | duration R2 | duration rho | size R2 | size rho | open after 24 h AUC | 100 acres AUC |
|---|---|---|---|---|---|---|
| N (class) | 0.055 | 0.218 | 0.091 | 0.303 | 0.623 | 0.674 |
| W | 0.009 | 0.092 | 0.042 | 0.198 | 0.542 | 0.633 |
| N+W | 0.065 | 0.239 | 0.132 | 0.364 | 0.632 | 0.715 |
| P (pixel) | 0.122 | 0.308 | 0.233 | 0.483 | 0.690 | 0.785 |
| E (150 m) | 0.129 | 0.320 | 0.244 | 0.493 | 0.695 | 0.795 |
| P+W | 0.127 | 0.316 | 0.270 | 0.526 | 0.693 | 0.804 |
| E+W | 0.134 | 0.327 | 0.281 | 0.535 | 0.697 | 0.814 |
| E+N+W | 0.134 | 0.328 | 0.284 | 0.537 | 0.697 | 0.813 |

The embedding alone beats the class with the weather on every target, and the class adds nothing on top
of the embedding. Size is the better-read target: the 150 m embedding with the hour's weather explains
28 percent of the variance in log size and ranks a 100-acre fire against a smaller one four times in
five. Duration is weaker (13 percent), as expected from an administrative out time. The weather matters
for size, not for duration.

`models/burn_duration.json` carries the pixel-embedding ridge heads (P+W) for both targets in the land
cover head's format (mean, scale, coef, intercept; the weather columns after the 64 bands), so the map
could draw expected size or duration per pixel the way it draws the ignition logit. Not wired in.

## Limits

- The out time is a reporting event. A fire "out" at 74 hours was likely out of flame well before.
  The ordering across classes is what the hold hours use, not the hours themselves as a burn time.
- The class is the prior-year NLCD at the ignition point; the embedding is the prior-year 10 m pixel
  (2025 fires take 2024 ground and labels, since there is no 2025 NLCD).
- Fires are the WFIGS points, so the same reporting gaps as the calibration check (the Southeast under).
- The heads are fit on fires only. They say how a fire that started here would go, not whether one starts.
