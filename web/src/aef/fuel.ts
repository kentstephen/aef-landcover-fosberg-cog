import type { Device, Texture } from "@luma.gl/core";
import { HEADS, type Teacher } from "./head.js";
import { hexToRgb } from "./ramp.js";

/**
 * How much of the hour's FFWI each land cover class carries, 0..1: the Fosberg index is written for
 * fine dead fuel (cured grass, litter, brush), so open herbaceous and shrub cover carry all of it,
 * forest carries part (the surface fuel under a canopy is shaded and damper, evergreen litter more
 * than deciduous), crops and bare ground little, built-up and water none. First cut; to be tuned by
 * looking at the map.
 */
export const FUEL_NLCD: Record<number, number> = {
  11: 0, 12: 0,
  21: 0.4, 22: 0.2, 23: 0, 24: 0,
  31: 0.25,
  41: 0.4, 42: 0.6, 43: 0.5,
  52: 1, 71: 1, 81: 0.9, 82: 0.5,
  90: 0.3, 95: 0.4,
};
export const FUEL_ESRI: Record<number, number> = {
  1: 0, 2: 0.5, 4: 0.35, 5: 0.5, 7: 0.15, 8: 0.25, 9: 0, 10: 0, 11: 1,
};

export const FUEL: Record<Teacher, Record<number, number>> = { nlcd: FUEL_NLCD, esri: FUEL_ESRI };

export function fuelOf(teacher: Teacher, classIndex: number): number {
  const code = HEADS[teacher].classes[classIndex];
  return code === undefined ? 0 : (FUEL[teacher][code] ?? 0);
}

/** One row per class index: rgb = the land cover color, a = the fuel weight x 255. 32 wide (16 NLCD classes fit). */
export const LUT_WIDTH = 32;

export function lutBytes(teacher: Teacher): Uint8Array {
  const h = HEADS[teacher];
  const out = new Uint8Array(LUT_WIDTH * 4);
  h.classes.forEach((code, i) => {
    const [r, g, b] = hexToRgb(h.colors[i]!);
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = Math.round((FUEL[teacher][code] ?? 0) * 255);
  });
  return out;
}

export function lutTexture(device: Device, teacher: Teacher): Texture {
  return device.createTexture({
    format: "rgba8unorm",
    width: LUT_WIDTH,
    height: 1,
    mipLevels: 1,
    data: lutBytes(teacher),
    sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
}
