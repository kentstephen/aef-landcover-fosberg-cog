import { GeoJsonLayer } from "@deck.gl/layers";
import type { Feature, MultiPolygon, Polygon } from "geojson";
import { OFFSET } from "./aef/calibration.js";
import type { CountyPart } from "./aef/coarse.js";
import { rampStripe, type GroundRamp, rampStops } from "./aef/ramp.js";
import type { WeatherSource } from "./aef/weather.js";
import { HRRR_SHARD } from "./hrrr/metadata.js";

/**
 * The pixel picture averaged by county. The coarse workers sum exp(ground logit) per county and HRRR pixel
 * over every file of the index at one overview level (coarse.worker.ts `county`); here the hour's weather
 * term is folded in per HRRR pixel, so a county's value is the mean of exp(ground + weather) over its
 * pixels: the mean odds, which in calibrated units is ignitions per area and time. The county shapes are
 * the Census 2023 cartographic boundaries (scripts/build_counties.py).
 */
export type CountyProps = { name: string; state: string; km2: number; bbox: [number, number, number, number] };
export type CountyFeature = Feature<Polygon | MultiPolygon, CountyProps> & { id: string };

/** A lon/lat grid over CONUS with the county index (1-based) under each cell, for the workers. */
export type UnitsGrid = { ids: Uint16Array; width: number; height: number; lon0: number; lat0: number; step: number };

const LON0 = -125.5;
const LAT0 = 50;
const STEP = 0.02;
const GW = Math.ceil((-66 - LON0) / STEP);
const GH = Math.ceil((LAT0 - 24) / STEP);
/** ln(1000 km2 x 24 h): mean odds per km2 per hour to ignitions per 1000 km2 per day, with the calibration offset. */
const TO_RATE = Math.exp(OFFSET + Math.log(1000 * 24));

export async function loadCounties(baseUrl: string): Promise<CountyFeature[]> {
  const r = await fetch(`${baseUrl}counties.json`);
  if (!r.ok) throw new Error(`counties: HTTP ${r.status}`);
  const j = (await r.json()) as { features: CountyFeature[] };
  return j.features;
}

/**
 * The county under each grid cell, drawn once to an offscreen canvas with the county index in the fill
 * color. Edge cells blend two counties' colors; those fail the check byte and take a neighbor's index.
 */
export function unitsGrid(features: CountyFeature[]): UnitsGrid {
  const canvas = new OffscreenCanvas(GW, GH);
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, GW, GH);
  const ring = (coords: number[][]) => {
    coords.forEach(([lon, lat], i) => {
      const x = (lon! - LON0) / STEP;
      const y = (LAT0 - lat!) / STEP;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
  };
  features.forEach((f, i) => {
    const k = i + 1;
    const r = k & 255;
    const g = k >> 8;
    ctx.fillStyle = `rgb(${r},${g},${255 - r})`;
    ctx.beginPath();
    const polys = f.geometry.type === "Polygon" ? [f.geometry.coordinates] : f.geometry.coordinates;
    for (const poly of polys) for (const rg of poly) ring(rg);
    ctx.fill("evenodd");
  });
  const img = ctx.getImageData(0, 0, GW, GH).data;
  const ids = new Uint16Array(GW * GH);
  const n = GW * GH;
  for (let i = 0; i < n; i++) {
    const r = img[i * 4]!;
    const g = img[i * 4 + 1]!;
    const b = img[i * 4 + 2]!;
    const k = r + g * 256;
    if (k === 0) continue;
    if (b === 255 - r && k <= features.length) ids[i] = k;
    else ids[i] = 0xffff; // a blended edge cell: filled from a neighbor below
  }
  for (let i = 0; i < n; i++) {
    if (ids[i] !== 0xffff) continue;
    const x = i % GW;
    const y = (i - x) / GW;
    let got = 0;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]] as const) {
      const xx = x + dx;
      const yy = y + dy;
      if (xx < 0 || yy < 0 || xx >= GW || yy >= GH) continue;
      const v = ids[yy * GW + xx]!;
      if (v !== 0 && v !== 0xffff) {
        got = v;
        break;
      }
    }
    ids[i] = got;
  }
  return { ids, width: GW, height: GH, lon0: LON0, lat0: LAT0, step: STEP };
}

/** One county's cells: the HRRR pixels its ground pixels fall in, the sum of exp(ground logit) and the count in each. */
type Cells = { hx: number[]; hy: number[]; sum: number[]; n: number[]; total: number };

export class CountyTable {
  readonly cells: Cells[];
  /** One HRRR pixel per shard a county touches, for asking the weather in. */
  private shardPix: Map<string, [number, number]>[];
  pixelKm2 = 0;
  filesDone = 0;
  /** Bumps when the sums change (a file landed). */
  version = 0;
  private hours = new Map<number, { v: number; wv: number; odds: Float32Array }>();

  constructor(readonly features: CountyFeature[]) {
    this.cells = features.map(() => ({ hx: [], hy: [], sum: [], n: [], total: 0 }));
    this.shardPix = features.map(() => new Map());
  }

  add(p: CountyPart): void {
    this.pixelKm2 = p.pixelKm2;
    for (let i = 0; i < p.unit.length; i++) {
      const c = this.cells[p.unit[i]! - 1];
      if (!c) continue;
      const hx = p.hx[i]!;
      const hy = p.hy[i]!;
      // the same HRRR pixel may come from two files (a zone seam): add to the existing cell
      let j = -1;
      for (let k = c.hx.length - 1; k >= 0 && k >= c.hx.length - 8; k--) if (c.hx[k] === hx && c.hy[k] === hy) { j = k; break; }
      if (j < 0) {
        c.hx.push(hx);
        c.hy.push(hy);
        c.sum.push(p.sum[i]!);
        c.n.push(p.n[i]!);
      } else {
        c.sum[j]! += p.sum[i]!;
        c.n[j]! += p.n[i]!;
      }
      c.total += p.n[i]!;
      const sk = `${Math.floor(hy / HRRR_SHARD.height)},${Math.floor(hx / HRRR_SHARD.width)}`;
      const sp = this.shardPix[p.unit[i]! - 1]!;
      if (!sp.has(sk)) sp.set(sk, [hx, hy]);
    }
    this.filesDone++;
    this.version++;
    this.hours.clear();
  }

  /** The HRRR pixels (one per shard) under the counties whose boxes meet a lon/lat view. */
  pixelsInView(view: [number, number, number, number]): { px: Int16Array; py: Int16Array } {
    const xs: number[] = [];
    const ys: number[] = [];
    this.features.forEach((f, i) => {
      const b = f.properties.bbox;
      if (b[2] < view[0] || b[0] > view[2] || b[3] < view[1] || b[1] > view[3]) return;
      for (const [x, y] of this.shardPix[i]!.values()) {
        xs.push(x);
        ys.push(y);
      }
    });
    return { px: Int16Array.from(xs), py: Int16Array.from(ys) };
  }

  /**
   * Every county's mean odds at one hour: sum over its cells of exp(ground) x exp(weather) over the pixels
   * with weather in hand; NaN where under half the county's pixels have weather. Cached per hour until
   * the sums or the weather change (weatherVersion).
   */
  hour(h: number, src: WeatherSource, weatherVersion: number): Float32Array {
    const hit = this.hours.get(h);
    if (hit && hit.v === this.version && hit.wv === weatherVersion) return hit.odds;
    const odds = new Float32Array(this.features.length).fill(NaN);
    if (src.at) {
      for (let i = 0; i < this.cells.length; i++) {
        const c = this.cells[i]!;
        let acc = 0;
        let cnt = 0;
        for (let k = 0; k < c.hx.length; k++) {
          const w = src.at(c.hx[k]!, c.hy[k]!, h);
          if (w !== w) continue;
          acc += c.sum[k]! * Math.exp(w);
          cnt += c.n[k]!;
        }
        if (c.total > 0 && cnt >= 0.5 * c.total) odds[i] = acc / cnt;
      }
    }
    this.hours.set(h, { v: this.version, wv: weatherVersion, odds });
    return odds;
  }

  /** The hour's odds as ignitions per 1000 km2 per day. */
  static rate(odds: number): number {
    return odds * TO_RATE;
  }
}

export type CountyScale = { kind: "probability" | "logit"; lo: number; hi: number };

/** Mean odds through the panel's scale to a ramp position 0..1, or -1 for hidden. */
function position(odds: number, sc: CountyScale): number {
  if (!(odds > 0)) return -1;
  const v = sc.kind === "probability" ? odds / (1 + odds) : Math.log(odds);
  const t = (v - sc.lo) / (sc.hi - sc.lo);
  return t < 0 ? -1 : Math.min(1, t);
}

export function countyLayer(o: {
  features: CountyFeature[];
  odds0: Float32Array;
  odds1: Float32Array | null;
  frac: number;
  scale: CountyScale;
  ramp: GroundRamp;
  opacity: number;
  visible: boolean;
  beforeId?: string;
  trigger: unknown[];
  onHover: (index: number, x: number, y: number) => void;
}): GeoJsonLayer<CountyProps> {
  const stripe = rampStripe(rampStops(o.ramp));
  const color = (i: number): [number, number, number, number] => {
    const a = o.odds0[i]!;
    const b = o.odds1 ? o.odds1[i]! : NaN;
    const odds = b === b && a === a ? a + (b - a) * o.frac : a === a ? a : b;
    const t = position(odds, o.scale);
    if (t < 0) return [0, 0, 0, 0];
    const x = Math.round(t * 255) * 4;
    // fades in over the first tenth of the ramp so the hide line is not a hard edge
    return [stripe[x]!, stripe[x + 1]!, stripe[x + 2]!, Math.round(255 * Math.min(1, t / 0.1))];
  };
  const index = new Map(o.features.map((f, i) => [f.id, i]));
  return new GeoJsonLayer<CountyProps>({
    id: "counties",
    data: o.features as never,
    filled: true,
    stroked: true,
    getFillColor: ((f: CountyFeature) => color(index.get(f.id)!)) as never,
    getLineColor: [255, 255, 255, 36],
    getLineWidth: 1,
    lineWidthUnits: "pixels",
    opacity: o.opacity,
    pickable: true,
    onHover: ((info: { object?: CountyFeature; x: number; y: number }) => o.onHover(info.object ? index.get(info.object.id)! : -1, info.x, info.y)) as never,
    visible: o.visible,
    beforeId: o.beforeId,
    updateTriggers: { getFillColor: o.trigger },
  } as never);
}
