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
 * How long each class stays lit after the weather eases, in hours: the ground holds a past hour's
 * peak, sliding back under the gate over the class's hold. From data (ignition/burn_head.py,
 * models/burn_duration.json): the median hours from discovery to the fire's out time for WFIGS fires
 * 2019 to 2026 that started in each NLCD class (17,701 fires with an out time), capped at four days.
 * Fires in timber, shrub and grass run three to four days by that clock, pasture and crops two, built
 * ground one to two (dense built a few hours), water and ice none. The out time is the agency's, so
 * it runs past the flames; it is the same clock for every class, so the ordering holds.
 */
export const HOLD_NLCD: Record<number, number> = {
  11: 0, 12: 0,
  21: 51, 22: 45, 23: 27, 24: 3,
  31: 63,
  41: 94, 42: 96, 43: 27,
  52: 95, 71: 90, 81: 45, 82: 45,
  90: 51, 95: 78,
};
/** The Esri classes carried over from their NLCD counterparts (trees 96, rangeland 93, flooded vegetation 65). */
export const HOLD_ESRI: Record<number, number> = {
  1: 0, 2: 96, 4: 65, 5: 45, 7: 40, 8: 63, 9: 0, 10: 0, 11: 93,
};
export const HOLD: Record<Teacher, Record<number, number>> = { nlcd: HOLD_NLCD, esri: HOLD_ESRI };
/** The hold hours are stored in the lookup texture as a byte, hours x HOLD_SCALE (so up to 127 h). */
export const HOLD_SCALE = 2;

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
