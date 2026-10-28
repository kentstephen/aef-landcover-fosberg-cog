import { GeoArrowPathLayer } from "@geoarrow/deck.gl-geoarrow";
import * as arrow from "apache-arrow";
import { HRRR_LEAD_TIME_COUNT } from "../hrrr/metadata.js";
import type { FfwiTileCache } from "../hrrr/tile-cache.js";
import { KM_TO_MI, type LineSet } from "./load.js";

/**
 * Transmission lines lit by the weather. One GeoArrowPathLayer over every
 * piece: the geometry is uploaded once from the Arrow table; per frame only
 * a color vector (RGBA per piece) and a width vector (px per piece) change,
 * both handed in as Arrow vectors, so deck expands them to per-vertex
 * attributes without calling back per path.
 *
 * Color and width both follow the same number, FFWI at the piece's pixel,
 * normalised from the hide line (t = 0) to the ramp top (t = 1): color from
 * the chosen colormap, width from 1 px up to `maxWidth`. The colormap is
 * sampled from RAMP_FLOOR upward, not from 0: the dark end of the warm
 * ramps (maroon on solar / inferno) at 1 px on a dark basemap is invisible
 * to a protan eye, and a lit line must read as lit. Below the hide line a
 * piece is a faint ghost at 1 px. Voltage class lives in the tooltip only,
 * so a fat line always means high FFWI.
 */

/** FFWI for every unique pixel and lead, computed once per cache change. */
export type LineRun = {
  /** [pixel * 49 + lead], NaN where not loaded. */
  series: Float32Array;
  /** Per-piece t in 0..1 (NaN = unloaded) is derived per frame from this. */
  milesLoaded: number;
};

export function lineRun(lines: LineSet, cache: FfwiTileCache): LineRun {
  const nPix = lines.pixelX.length;
  const series = new Float32Array(nPix * HRRR_LEAD_TIME_COUNT);
  const loaded = new Uint8Array(nPix);
  const tmp = new Float32Array(HRRR_LEAD_TIME_COUNT);
  for (let j = 0; j < nPix; j++) {
    cache.ffwiSeries(lines.pixelX[j]!, lines.pixelY[j]!, tmp);
    series.set(tmp, j * HRRR_LEAD_TIME_COUNT);
    loaded[j] = Number.isFinite(tmp[0]!) ? 1 : 0;
  }
  let loadedKm = 0;
  for (let i = 0; i < lines.n; i++) if (loaded[lines.pieceToPixel[i]!]) loadedKm += lines.km[i]!;
  return { series, milesLoaded: loadedKm * KM_TO_MI };
}

/** Miles of line with FFWI >= hideBelow per lead (whole run). */
export function milesAbove(lines: LineSet, run: LineRun, hideBelow: number): Float64Array {
  const out = new Float64Array(HRRR_LEAD_TIME_COUNT);
  for (let i = 0; i < lines.n; i++) {
    const base = lines.pieceToPixel[i]! * HRRR_LEAD_TIME_COUNT;
    const km = lines.km[i]!;
    for (let l = 0; l < HRRR_LEAD_TIME_COUNT; l++) if (run.series[base + l]! >= hideBelow) out[l] += km;
  }
  for (let l = 0; l < HRRR_LEAD_TIME_COUNT; l++) out[l] *= KM_TO_MI;
  return out;
}

/** FFWI per piece at one lead, from the run table. */
export function lineFfwiAt(lines: LineSet, run: LineRun, lead: number): Float32Array {
  const out = new Float32Array(lines.n);
  for (let i = 0; i < lines.n; i++) out[i] = run.series[lines.pieceToPixel[i]! * HRRR_LEAD_TIME_COUNT + lead]!;
  return out;
}

/** Lit lines use the colormap from this fraction up (see the header). */
export const RAMP_FLOOR = 0.35;

function sampleColormap(img: ImageData, row: number, v: number): [number, number, number] {
  const x = Math.max(0, Math.min(255, Math.round(v * 255)));
  const i = (row * img.width + x) * 4;
  return [img.data[i]!, img.data[i + 1]!, img.data[i + 2]!];
}

/** deck.gl-geoarrow 0.4 takes accessors as arrow.Data (an arrow.Vector is silently ignored). */
function rgbaData(colors: Uint8Array, n: number): arrow.Data<arrow.FixedSizeList<arrow.Uint8>> {
  const child = arrow.makeData({ type: new arrow.Uint8(), length: n * 4, data: colors });
  return arrow.makeData({
    type: new arrow.FixedSizeList(4, new arrow.Field("rgba", new arrow.Uint8())),
    length: n,
    nullCount: 0,
    child,
  });
}

export type LineLayerArgs = {
  lines: LineSet;
  /** FFWI per piece at this lead (lineFfwiAt). */
  ffwi: Float32Array;
  hideBelow: number;
  ffwiMax: number;
  colormap: ImageData;
  colormapIndex: number;
  /** Opacity of pieces below the line (0 hides them). */
  ghostAlpha: number;
  /** Width in px at the ramp top; 1 px at the hide line. */
  maxWidth: number;
  onHover: (index: number | null, x: number, y: number) => void;
  beforeId?: string;
};

export function buildLineLayer(a: LineLayerArgs) {
  const { lines } = a;
  const n = lines.n;
  const colors = new Uint8Array(n * 4);
  const widths = new Float32Array(n);
  const ghost = Math.round(255 * a.ghostAlpha);
  const span = Math.max(1, a.ffwiMax - a.hideBelow);
  // 256-entry lookup of the ramp, so the per-piece loop is arithmetic only.
  const lut = new Uint8Array(256 * 3);
  for (let k = 0; k < 256; k++) {
    const [r, g, b] = sampleColormap(a.colormap, a.colormapIndex, RAMP_FLOOR + (1 - RAMP_FLOOR) * (k / 255));
    lut[k * 3] = r;
    lut[k * 3 + 1] = g;
    lut[k * 3 + 2] = b;
  }
  for (let i = 0; i < n; i++) {
    const v = a.ffwi[i]!;
    const o = i * 4;
    if (!(v >= a.hideBelow)) {
      colors[o] = 120;
      colors[o + 1] = 135;
      colors[o + 2] = 160;
      colors[o + 3] = ghost;
      widths[i] = 1;
      continue;
    }
    const t = Math.min(1, (v - a.hideBelow) / span);
    const k = Math.round(t * 255) * 3;
    colors[o] = lut[k]!;
    colors[o + 1] = lut[k + 1]!;
    colors[o + 2] = lut[k + 2]!;
    colors[o + 3] = 255;
    widths[i] = 1 + (a.maxWidth - 1) * t;
  }
  return new GeoArrowPathLayer({
    id: "transmission-lines",
    data: lines.table,
    getColor: rgbaData(colors, n),
    getWidth: arrow.makeData({ type: new arrow.Float32(), length: n, data: widths }),
    widthUnits: "pixels",
    widthMinPixels: 1,
    jointRounded: false,
    capRounded: false,
    pickable: true,
    onHover: (info: { index: number; x: number; y: number }) => a.onHover(info.index >= 0 ? info.index : null, info.x, info.y),
    parameters: { depthCompare: "always" } as never,
    beforeId: a.beforeId,
  } as never);
}
