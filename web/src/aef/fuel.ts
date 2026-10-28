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

/**
 * How long each class stays lit after the weather eases, in hours (the afterglow's time constant):
 * fine fuels (grass, pasture, shrub) answer the hour and let go of it fast; timber holds a dry spell
 * for a day; built ground and water hold nothing. The lit value at an hour is the brightest of the
 * hours behind it, each faded by exp(-age / hold). A picture of what the window has done to a
 * place, not only what this hour does.
 */
export const HOLD_NLCD: Record<number, number> = {
  11: 0, 12: 0,
  21: 3, 22: 2, 23: 0, 24: 0,
  31: 2,
  41: 18, 42: 24, 43: 20,
  52: 6, 71: 3, 81: 3, 82: 3,
  90: 12, 95: 6,
};
export const HOLD_ESRI: Record<number, number> = {
  1: 0, 2: 20, 4: 10, 5: 4, 7: 2, 8: 2, 9: 0, 10: 0, 11: 6,
};
export const HOLD: Record<Teacher, Record<number, number>> = { nlcd: HOLD_NLCD, esri: HOLD_ESRI };
/** The hold hours are stored in the lookup texture as a byte, hours x HOLD_SCALE. */
export const HOLD_SCALE = 4;

export function fuelOf(teacher: Teacher, classIndex: number): number {
  const code = HEADS[teacher].classes[classIndex];
  return code === undefined ? 0 : (FUEL[teacher][code] ?? 0);
}

/**
 * One column per class index, two rows: row 0 rgb = the land cover color, a = the fuel weight x 255;
 * row 1 r = the hold hours x HOLD_SCALE. 32 wide (16 NLCD classes fit).
 */
export const LUT_WIDTH = 32;
export const LUT_HEIGHT = 2;

export function lutBytes(teacher: Teacher): Uint8Array {
  const h = HEADS[teacher];
  const out = new Uint8Array(LUT_WIDTH * LUT_HEIGHT * 4);
  h.classes.forEach((code, i) => {
    const [r, g, b] = hexToRgb(h.colors[i]!);
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = Math.round((FUEL[teacher][code] ?? 0) * 255);
    const o = (LUT_WIDTH + i) * 4;
    out[o] = Math.min(255, Math.round((HOLD[teacher][code] ?? 0) * HOLD_SCALE));
    out[o + 3] = 255;
  });
  return out;
}

export function lutTexture(device: Device, teacher: Teacher): Texture {
  return device.createTexture({
    format: "rgba8unorm",
    width: LUT_WIDTH,
    height: LUT_HEIGHT,
    mipLevels: 1,
    data: lutBytes(teacher),
    sampler: { minFilter: "nearest", magFilter: "nearest", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
}
