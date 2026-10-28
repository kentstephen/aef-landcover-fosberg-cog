import { cellToLatLng, polygonToCells } from "h3-js";
import { lonLatToHrrrPixel } from "../hrrr/metadata.js";
import { HRRR_LEAD_TIME_COUNT } from "../hrrr/metadata.js";
import type { FfwiTileCache } from "../hrrr/tile-cache.js";
import { samplePotential, type PotentialRaster } from "../potential/service.js";

/**
 * Counties as the unit of display. Geometry: the US Census counties GeoJSON
 * that plotly mirrors (3.2 MB, CORS *, FIPS `id`, NAME/STATE props); Alaska,
 * Hawaii and Puerto Rico dropped. The join is H3 under the hood: each county
 * is polyfilled once at res 6 (~210k cells for CONUS, under a second), each
 * cell centroid becomes an HRRR pixel, and the per-county pixel lists are
 * what every frame reduces over. Counties too small for a res-6 cell (14 of
 * them) fall back to the pixel under their centroid.
 */
const COUNTIES_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json";
const SKIP_STATES = new Set(["02", "15", "72"]);
const JOIN_RES = 6;

export const STATE_ABBR: Record<string, string> = {
  "01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL",
  "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME",
  "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV",
  "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR",
  "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA",
  "53": "WA", "54": "WV", "55": "WI", "56": "WY",
};

export type CountyFeature = {
  type: "Feature";
  id: string;
  properties: { NAME: string; STATE: string; COUNTY: string; LSAD?: string; CENSUSAREA?: number };
  geometry: { type: "Polygon" | "MultiPolygon"; coordinates: number[][][] | number[][][][] };
};

export type CountySet = {
  features: CountyFeature[];
  name: string[];
  state: string[];
  /** Flattened per-county pixel lists: county i owns [pxStart[i], pxStart[i+1]). */
  pxStart: Int32Array;
  pxX: Int16Array;
  pxY: Int16Array;
  /** Lon/lat of the sample points (cell centroids), same indexing as pxX/pxY. */
  lon: Float64Array;
  lat: Float64Array;
};

export async function loadCounties(signal?: AbortSignal): Promise<CountySet> {
  const resp = await fetch(COUNTIES_URL, { signal });
  if (!resp.ok) throw new Error(`counties geojson HTTP ${resp.status}`);
  const gj = (await resp.json()) as { features: CountyFeature[] };
  const features = gj.features.filter((f) => !SKIP_STATES.has(f.properties.STATE));
  features.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));

  const n = features.length;
  const pxStart = new Int32Array(n + 1);
  const xs: number[] = [];
  const ys: number[] = [];
  const lons: number[] = [];
  const lats: number[] = [];
  const name: string[] = [];
  const state: string[] = [];
  for (let i = 0; i < n; i++) {
    const f = features[i]!;
    name.push(f.properties.NAME);
    state.push(STATE_ABBR[f.properties.STATE] ?? f.properties.STATE);
    const g = f.geometry;
    const polys = (g.type === "MultiPolygon" ? g.coordinates : [g.coordinates]) as number[][][][];
    const seen = new Set<number>();
    let cx = 0;
    let cy = 0;
    let cn = 0;
    for (const poly of polys) {
      let ids: string[] = [];
      try {
        ids = polygonToCells(poly, JOIN_RES, true);
      } catch {
        try {
          ids = polygonToCells([poly[0]!], JOIN_RES, true);
        } catch {
          /* leave empty; centroid fallback below */
        }
      }
      for (const id of ids) {
        const [la, lo] = cellToLatLng(id);
        const px = lonLatToHrrrPixel(lo, la);
        if (!px) continue;
        const key = px.y * 2048 + px.x;
        if (seen.has(key)) continue;
        seen.add(key);
        xs.push(px.x);
        ys.push(px.y);
        lons.push(lo);
        lats.push(la);
      }
      for (const v of poly[0]!) {
        cx += v[0]!;
        cy += v[1]!;
        cn++;
      }
    }
    if (seen.size === 0 && cn > 0) {
      const lo = cx / cn;
      const la = cy / cn;
      const px = lonLatToHrrrPixel(lo, la);
      if (px) {
        xs.push(px.x);
        ys.push(px.y);
        lons.push(lo);
        lats.push(la);
      }
    }
    pxStart[i + 1] = xs.length;
  }
  return {
    features,
    name,
    state,
    pxStart,
    pxX: Int16Array.from(xs),
    pxY: Int16Array.from(ys),
    lon: Float64Array.from(lons),
    lat: Float64Array.from(lats),
  };
}

/** Mean potential (display units) over each county's sample points; NaN where none. Static per potential layer. */
export function countyPotentialMean(cs: CountySet, potential: PotentialRaster): Float32Array {
  const n = cs.features.length;
  const out = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let k = 0;
    for (let j = cs.pxStart[i]!; j < cs.pxStart[i + 1]!; j++) {
      const v = samplePotential(potential, cs.lon[j]!, cs.lat[j]!);
      if (Number.isFinite(v)) {
        sum += v;
        k++;
      }
    }
    if (k > 0) out[i] = sum / k;
  }
  return out;
}

export type CountyFrame = {
  /** Mean / max FFWI over the county's pixels this hour; NaN if none loaded. */
  mean: Float32Array;
  max: Float32Array;
  /** Share (0..1) of the county's pixels with FFWI >= threshold this hour. */
  share: Float32Array;
  /** Mean over pixels of the number of hours in the trailing window (ending this hour) with FFWI >= threshold. */
  hours: Float32Array;
  /** Hours the window could actually see (min(window, lead + 1)). */
  windowSeen: number;
  loadedCounties: number;
};

export function countyFrame(
  cs: CountySet,
  cache: FfwiTileCache,
  lead: number,
  threshold: number,
  windowHours: number,
): CountyFrame {
  const n = cs.features.length;
  const mean = new Float32Array(n).fill(NaN);
  const max = new Float32Array(n).fill(NaN);
  const share = new Float32Array(n).fill(NaN);
  const hours = new Float32Array(n).fill(NaN);
  const firstLead = Math.max(0, lead - windowHours + 1);
  const windowSeen = lead - firstLead + 1;
  let loaded = 0;
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let mx = -Infinity;
    let k = 0;
    let above = 0;
    let hrs = 0;
    for (let j = cs.pxStart[i]!; j < cs.pxStart[i + 1]!; j++) {
      const x = cs.pxX[j]!;
      const y = cs.pxY[j]!;
      const f = cache.ffwiAt(x, y, lead);
      if (!Number.isFinite(f)) continue;
      sum += f;
      if (f > mx) mx = f;
      if (f >= threshold) above++;
      k++;
      for (let l = firstLead; l < lead; l++) if (cache.ffwiAt(x, y, l) >= threshold) hrs++;
      if (f >= threshold) hrs++;
    }
    if (k === 0) continue;
    loaded++;
    mean[i] = sum / k;
    max[i] = mx;
    share[i] = above / k;
    hours[i] = hrs / k;
  }
  return { mean, max, share, hours, windowSeen, loadedCounties: loaded };
}

/**
 * Over the whole 48 h: per county, the lead hour at which its mean FFWI peaks
 * (and the peak value), and the first lead at which the mean crosses the
 * threshold (-1 if never). Uses the cache's current timelag. ~10 M pixel-hours
 * for CONUS, a few hundred ms; callers memoize per cache version.
 */
export type CountyPeak = {
  peakLead: Int8Array;
  peakValue: Float32Array;
  onsetLead: Int8Array;
  /** Mean FFWI per county per lead, lead-major: idx = lead * n + county. */
  meanByLead: Float32Array;
  /** Share (0..1) of the county's pixels with FFWI >= threshold, per lead, same indexing. */
  shareByLead: Float32Array;
};

export function countyPeak(cs: CountySet, cache: FfwiTileCache, threshold: number): CountyPeak {
  const n = cs.features.length;
  const L = HRRR_LEAD_TIME_COUNT;
  const sums = new Float64Array(L);
  const counts = new Int32Array(L);
  const above = new Int32Array(L);
  const series = new Float32Array(L);
  const peakLead = new Int8Array(n).fill(-1);
  const peakValue = new Float32Array(n).fill(NaN);
  const onsetLead = new Int8Array(n).fill(-1);
  const meanByLead = new Float32Array(L * n).fill(NaN);
  const shareByLead = new Float32Array(L * n).fill(NaN);
  for (let i = 0; i < n; i++) {
    sums.fill(0);
    counts.fill(0);
    above.fill(0);
    for (let j = cs.pxStart[i]!; j < cs.pxStart[i + 1]!; j++) {
      cache.ffwiSeries(cs.pxX[j]!, cs.pxY[j]!, series);
      for (let l = 0; l < L; l++) {
        const f = series[l]!;
        if (f === f) {
          sums[l] += f;
          counts[l]++;
          if (f >= threshold) above[l]++;
        }
      }
    }
    let best = -Infinity;
    for (let l = 0; l < L; l++) {
      if (counts[l] === 0) continue;
      const m = sums[l]! / counts[l]!;
      meanByLead[l * n + i] = m;
      shareByLead[l * n + i] = above[l]! / counts[l]!;
      if (m > best) {
        best = m;
        peakLead[i] = l;
        peakValue[i] = m;
      }
      if (onsetLead[i] < 0 && m >= threshold) onsetLead[i] = l;
    }
  }
  return { peakLead, peakValue, onsetLead, meanByLead, shareByLead };
}


