import { GeoJsonLayer } from "@deck.gl/layers";
import type { CountyFeature, CountyFrame, CountyPeak, CountySet } from "./load.js";
export type { CountyField, CountyStat, PeakMode } from "../ui/ControlPanel.js";
import type { CountyField, CountyStat, PeakMode } from "../ui/ControlPanel.js";

/**
 * One choropleth, three states. "peak": the hour (of the 48) at which the
 * county's mean FFWI peaks, or first crosses the hide line; a time map on a
 * sequential ramp, static. "ffwi": each county colored by a statistic of
 * FFWI over its pixels this hour (mean, max, or the share of pixels above
 * the hide line, or hours above it in a trailing window), animated. "bp": the same polygons colored by the county's
 * mean burn probability, static, on a sqrt ramp. Only one is ever shown;
 * the other is in the tooltip.
 */

export type CountyLayerArgs = {
  counties: CountySet;
  frame: CountyFrame;
  bpMean: Float32Array;
  /** Whole-run peak/onset per county; null until computed (field "peak"). */
  peak: CountyPeak | null;
  peakMode: PeakMode;
  field: CountyField;
  stat: CountyStat;
  colormap: ImageData;
  colormapIndex: number;
  hideBelow: number;
  ffwiMax: number;
  bpFull: number;
  /** Trailing window (hours) for the "hours" stat. */
  windowHours: number;
  opacity: number;
  updateKey: unknown[];
  onHover: (index: number | null, x: number, y: number) => void;
  onClick: (index: number | null) => void;
  beforeId?: string;
};

function sampleColormap(img: ImageData, row: number, v: number): [number, number, number] {
  const x = Math.max(0, Math.min(255, Math.round(v * 255)));
  const i = (row * img.width + x) * 4;
  return [img.data[i]!, img.data[i + 1]!, img.data[i + 2]!];
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Normalized 0..1 value per county for the current field/stat, NaN = not drawn. */
export function countyValue(a: CountyLayerArgs, i: number): number {
  if (a.field === "peak") {
    if (!a.peak) return NaN;
    if (a.peakMode === "onset") {
      const l = a.peak.onsetLead[i]!;
      return l >= 0 ? l / 48 : NaN;
    }
    const l = a.peak.peakLead[i]!;
    return l >= 0 && a.peak.peakValue[i]! >= a.hideBelow ? l / 48 : NaN;
  }
  if (a.field === "bp") {
    const b = a.bpMean[i]!;
    return Number.isFinite(b) ? Math.sqrt(clamp01(b / a.bpFull)) : NaN;
  }
  if (a.stat === "share") {
    const s = a.frame.share[i]!;
    return Number.isFinite(s) && s > 0 ? s : NaN;
  }
  if (a.stat === "hours") {
    const h = a.frame.hours[i]!;
    return Number.isFinite(h) && h > 0 ? clamp01(h / a.windowHours) : NaN;
  }
  const v = (a.stat === "mean" ? a.frame.mean : a.frame.max)[i]!;
  if (!Number.isFinite(v) || v < a.hideBelow) return NaN;
  return clamp01((v - a.hideBelow) / Math.max(1, a.ffwiMax - a.hideBelow));
}

export function buildCountyLayer(a: CountyLayerArgs) {
  const n = a.counties.features.length;
  const colors = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const v = countyValue(a, i);
    if (!Number.isFinite(v)) continue;
    const [r, g, b] = sampleColormap(a.colormap, a.colormapIndex, v);
    colors[i * 4] = r;
    colors[i * 4 + 1] = g;
    colors[i * 4 + 2] = b;
    colors[i * 4 + 3] = 255;
  }
  return new GeoJsonLayer<CountyFeature>({
    id: "counties",
    data: a.counties.features,
    filled: true,
    stroked: false,
    getFillColor: (_f: CountyFeature, { index }: { index: number }) =>
      [colors[index * 4]!, colors[index * 4 + 1]!, colors[index * 4 + 2]!, colors[index * 4 + 3]!] as [number, number, number, number],
    opacity: a.opacity,
    pickable: true,
    autoHighlight: true,
    highlightColor: [255, 255, 255, 70],
    onHover: (info: { index: number; x: number; y: number }) => a.onHover(info.index >= 0 ? info.index : null, info.x, info.y),
    onClick: (info: { index: number }) => a.onClick(info.index >= 0 ? info.index : null),
    updateTriggers: { getFillColor: a.updateKey },
    beforeId: a.beforeId,
  } as never);
}
