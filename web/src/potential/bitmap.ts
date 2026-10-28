import { POTENTIAL_LAYERS, type PotentialRaster } from "./service.js";

/**
 * Paint the potential raster into an ImageData for a deck.gl BitmapLayer.
 * Cool single-hue ramp (pale cyan -> deep blue), alpha rising with value,
 * so it stays distinct from the warm FFWI ramp and never encodes anything
 * in red vs green.
 */
export function potentialToImageData(
  r: PotentialRaster,
  displayMax: number,
): ImageData {
  const img = new ImageData(r.width, r.height);
  const d = img.data;
  const scale = POTENTIAL_LAYERS[r.layer].scale;
  const lo = [224, 247, 255]; // pale cyan
  const hi = [8, 48, 140]; // deep blue
  for (let i = 0; i < r.width * r.height; i++) {
    const raw = r.values[i] ?? 0;
    const valid = !(r.mask && r.mask[i] === 0) && raw > 0;
    if (!valid) {
      d[i * 4 + 3] = 0;
      continue;
    }
    const t = Math.min(1, (raw * scale) / displayMax);
    const tt = Math.sqrt(t);
    d[i * 4] = Math.round(lo[0]! + (hi[0]! - lo[0]!) * tt);
    d[i * 4 + 1] = Math.round(lo[1]! + (hi[1]! - lo[1]!) * tt);
    d[i * 4 + 2] = Math.round(lo[2]! + (hi[2]! - lo[2]!) * tt);
    // Near-invisible at the low end so the layer reads as "where it can burn",
    // not as a land mask.
    d[i * 4 + 3] = Math.round(255 * Math.pow(t, 0.6));
  }
  return img;
}
