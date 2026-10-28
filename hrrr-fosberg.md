# HRRR + Fosberg Fire Weather Index

## HRRR

NOAA's High-Resolution Rapid Refresh. A 3 km convection-allowing model covering CONUS, updated hourly.

dynamical.org republishes it as cloud-optimized Icechunk Zarr — open directly with Xarray, no GRIB handling.

| | |
|---|---|
| Dataset | `noaa-hrrr-analysis` |
| Resolution | 3 km, hourly |
| Coverage | CONUS |
| Time span | 2014-10-01 to present |
| Projection | Lambert Conformal Conic, indexed on `x` / `y` |

Catalog page: https://dynamical.org/catalog/noaa-hrrr-analysis/

There is also an 18-hour and 48-hour forecast version if you want lead time instead of past conditions.

### Variables needed for Fosberg

| Variable | Units |
|---|---|
| `temperature_2m` | °C |
| `relative_humidity_2m` | % |
| `wind_u_10m` | m/s |
| `wind_v_10m` | m/s |
| `wind_gust_surface` | m/s |

All four live in the same dataset on the same grid at the same timestep, so no cross-source join is required for the weather layer.

Note: `relative_humidity_2m` is unavailable before August 2016.

---

## Fosberg Fire Weather Index

A fire-danger score computed from three weather values: temperature, relative humidity, and wind speed. No fuel map or drought history required.

It responds to instantaneous conditions rather than seasonal dryness, which makes it suited to short dangerous windows — dry frontal passages, downslope wind events — rather than long-term fire season severity.

### Formula

Fosberg works internally in °F and mph, so convert first.

**1. Equilibrium moisture content** — piecewise on relative humidity `h` (%) and temperature `T` (°F):

```
h < 10        m = 0.03229 + 0.281073·h − 0.000578·h·T
10 ≤ h < 50   m = 2.22749 + 0.160107·h − 0.014784·T
h ≥ 50        m = 21.0606 + 0.005565·h² − 0.00035·h·T − 0.483199·h
```

Clip `m` at 30, then scale: `mr = m / 30`

**2. Moisture damping coefficient**

```
η = 1 − 2·mr + 1.5·mr² − 0.5·mr³
```

**3. Index**

```
FFWI = η · √(1 + U²) / 0.3002
```

where `U` is wind speed in mph.

### Reading the output

The scale is anchored so that zero moisture with 30 mph wind returns 100. Values above that are possible and are usually left untruncated rather than capped.

The `√(1 + U²)` term is what makes wind the dominant escalating factor once humidity drops — which is the behavior you want if the question is "what happens when it gets windy."

### Caveats

- Fine fuels only. Assumes a fixed fuel bed (surface-area-to-volume ratio 3000 ft⁻¹, moisture of extinction 30%), so it says nothing about fuel type or load.
- Intended for hourly computation.
- Sustained wind is the standard input. Substituting gusts gives a more alarming number that is not comparable to published thresholds — if you use gusts, treat it as a separate field rather than as FFWI.
- Pair with a slower dryness layer (VIIRS vegetation health, SMAP soil moisture, or ESI) if you need seasonal context. Fosberg alone will read the same on a wet spring day and a drought day given identical temperature, humidity, and wind.

---

## Sources

- HRRR analysis — https://dynamical.org/catalog/noaa-hrrr-analysis/
- Fosberg (1978), *Weather in wildland fire management: the Fire Weather Index* — https://www.frames.gov/catalog/27085
- Goodrick (2002), modified FFWI including drought — https://www.srs.fs.usda.gov/pubs/ja/ja_goodrick001.pdf
- FPA-FOD ignition records, 7th edition — https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.7
