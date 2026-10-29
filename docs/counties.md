# The counties view

Branch `counties`. The pixel picture averaged by county, live in the browser, with a ranked table of the
counties in view. Nothing precomputed: the same AlphaEarth COG overviews the zoomed-out reader uses, at one
fixed level, summed per county and HRRR pixel in the coarse workers; the hour's weather folded in on the main
thread.

## How it works

- `scripts/build_counties.py`: Census 2023 cartographic boundaries (1:5m), the 48 states and DC, as
  `web/public/counties.json` (3,109 counties, 2.8 MB) with name, state, land area and bounding box.
- `web/src/counties.ts`: the county under each cell of a 0.02 degree lon/lat grid, drawn once to an offscreen
  canvas (index in the fill color; blended edge cells take a neighbor), sent to every coarse worker.
- `coarse.worker.ts` `county`: for one file at the 32 px level (2.56 km pixels, one range request a file), every
  pixel with ground is counted into a 16-bin histogram of its ground logit in its (county, HRRR pixel) cell.
  1,982 files; measured 109 s cold for the whole country, headless.
- `CountyTable.hour(h, zHide)`: per county, over the cells whose weather is in hand, the share of pixels the raster
  lights (the weather term at or over the gate, and ground + weather over the hide line), and the mean odds exp(ground +
  weather). NaN under half coverage; cached per hour and hide line. The SHARE drives the color (0 clear, all
  lit at the ramp top). The mean odds give the count: rate = mean odds x exp(calibration offset) x 24,000 =
  ignitions per 1000 km2 per day; starts today = rate x the county's land area / 1000.
- Why the share and not the mean odds: the mean of odds belongs to a county's hottest pixels (exp is convex),
  so a county with five percent of its ground at long odds read as lit while the raster showed only that five
  percent; the whole East lit up, Maine included. The share is the raster averaged.
- Counties under 25 ground pixels (about 160 km2) get no value: Virginia's independent cities are a few pixels of
  developed ground and read 100 percent lit at night.
- The weather under the counties in view is asked for through the same side channel as the tiles (one HRRR
  pixel per shard a county touches), so a county's number appears when its shards land, like the pixels.
- Drawn as a deck.gl GeoJsonLayer with the ground's ramp and scale (probability or logit), no outlines;
  the fill fades in over the first tenth of the ramp. Between hours the odds interpolate
  with the play fraction. Hover: county, rate, multiple of the CONUS mean, starts expected today. The table
  ranks the counties in view by rate; click a row to fly to the county.

## What it shows so far (first look, August 6, 2026 04Z, the West)

With the mean odds driving the color (since replaced by the share lit), small urban counties topped the rate list: Denver, Broomfield, Salt Lake, Oklahoma. This is the head's known
reading of developed open ground (the calibration check found the head over by about 4x on dense built and
cropland, and the county numbers are not class-corrected). Rate per area favors a small county with one bright
class; "starts today" favors the big ones (Stevens WA 0.24, Ferry 0.19). Neither is yet the county against
its own normal.

## Open

- The number that drives the shading: rate per area (now), expected starts, or the county against its own
  WFIGS normal for the month (needs each county's base rate from the record; the fire points are in
  data/ignition/wfigs.parquet). The ranking is sensitive to this choice.
- The class correction (observed/expected per class) is applied in the pixel tooltip only; the county sums
  carry the raw head, so urban counties read high.
- Coverage: a county whose shards are not in hand draws nothing, the same as one under the hide line. A
  "waiting" hatch or outline would tell them apart.
- Units: Overture division areas (counties, and localities within them) streamed as PMTiles instead of one
  GeoJSON; the shading is already on the GPU (deck.gl tessellates the polygons once; the hour only changes
  a color attribute), so what PMTiles buys is the first paint and finer units by zoom, not the per-hour draw.
  The sums still need a county id per pixel, which comes from the same polygons.
- Overview-pixel bias: an overview pixel is the mean embedding, so the head gives the logit of the mean, a
  little calmer than the pixels' mean odds where ground is mixed. Smallest in the uniform West.

## What would make it useful (pending, for the public and for responders)

- Public: one number per county today, against its own normal, and the hour it peaks. The pixel map as the
  detail behind a click, not the first thing seen.
- Responders: dispatch zones or Predictive Service Areas as the unit; expected starts and expected large fires
  (the size head, models/burn_duration.json, ranks a 100-acre fire at AUC 0.81); the forecast's disagreement
  between yesterday's and today's run as an uncertainty field; a ranked list that covers the whole country,
  which needs the whole country's shards (3.4 GB a week) or a coarser weather read for the list.
- Both: the head says where a fire could start, not where it goes; nothing here is spread.
