import type { Device, Texture } from "@luma.gl/core";

/**
 * Viridis as 50 stops: matplotlib's path (dark purple, blue, teal, green, yellow) with five darker
 * stops below its first color toward near-black and three paler stops above its last toward a
 * pale lemon, so the ramp has more range at both ends than the stock 256-entry table. Dark is calm,
 * bright is extreme. Interpolated into a 256-texel stripe for the GPU and a CSS gradient for the panel.
 */
export const VIRIDIS_STOPS = [
  "#07020f", "#13021d", "#1f022b", "#2c0138", "#380146", "#440154", "#460a5d", "#471365", "#481b6d", "#482374",
  "#472c7a", "#46337f", "#443a83", "#424186", "#3e4989", "#3c508b", "#39568c", "#365d8d", "#32648e", "#306a8e",
  "#2d708e", "#2b758e", "#287c8e", "#26828e", "#24878e", "#228d8d", "#20938c", "#1f998a", "#1f9f88", "#20a486",
  "#25ab82", "#2ab07f", "#32b67a", "#3bbb75", "#48c16e", "#54c568", "#60ca60", "#6ece58", "#7fd34e", "#8ed645",
  "#9dd93b", "#addc30", "#c0df25", "#d0e11c", "#dfe318", "#efe51c", "#fde725", "#feef54", "#fef784", "#ffffb3",
];

/**
 * Inferno (matplotlib) from 14 percent up, so the ramp starts on a dark violet that still shows against a
 * dark basemap instead of Inferno's black, with three paler stops above its pale yellow. Fiery: violet,
 * crimson, orange, yellow; luminance rises the whole way, no green.
 */
export const INFERNO_STOPS = [
  "#260c51", "#2f0a5b", "#380962", "#400a67", "#470b6a", "#4f0d6c", "#57106e", "#5f136e", "#67166e", "#6d186e",
  "#751b6e", "#7d1e6d", "#85216b", "#8d2369", "#932667", "#9b2964", "#a32c61", "#ab2f5e", "#b1325a", "#b93556",
  "#c03a51", "#c73e4c", "#ce4347", "#d34743", "#d94d3d", "#df5337", "#e45a31", "#e9612b", "#ec6726", "#f06f20",
  "#f37819", "#f68013", "#f8870e", "#fa9008", "#fb9906", "#fca309", "#fcac11", "#fcb418", "#fbbe23", "#f9c72f",
  "#f7d13d", "#f5db4c", "#f3e35a", "#f1ec6d", "#f2f482", "#f6fa96", "#fcffa4", "#fdffb0", "#fdffbc", "#feffc8",
];

/** Solar (cmocean) from 4 percent up (dark brown-red, orange, yellow) with three paler stops above. */
export const SOLAR_STOPS = [
  "#40181d", "#461a1f", "#4c1c20", "#541e22", "#5a2023", "#602124", "#682325", "#6e2525", "#742724", "#7b2a24",
  "#812c22", "#883021", "#8d331f", "#92361e", "#973b1c", "#9b401a", "#9f4419", "#a44a18", "#a84e16", "#ab5315",
  "#af5914", "#b25e14", "#b66313", "#b96a13", "#bc6f13", "#bf7413", "#c27b14", "#c48015", "#c78616", "#c98c18",
  "#cb921a", "#ce981c", "#d09f1e", "#d2a520", "#d4ab23", "#d6b226", "#d7b829", "#d9be2c", "#dac62f", "#dbcc32",
  "#ddd236", "#deda39", "#dfe13d", "#dfe740", "#e0f044", "#e0f647", "#e1fd4b", "#e7fe6b", "#eefe8b", "#f5ffab",
];

/** Fire (colorcet, linear_kryw_0_100_c71) from 10 percent up: deep red, orange, yellow, white. Its top is white already. */
export const FIRE_STOPS = [
  "#400000", "#480100", "#4f0100", "#550100", "#5d0100", "#650100", "#6b0200", "#740200", "#7c0200", "#830200",
  "#8b0300", "#930300", "#9c0400", "#a30400", "#ac0500", "#b40600", "#bc0700", "#c50800", "#ce0900", "#d50b00",
  "#de0d00", "#e71000", "#ef1700", "#f42000", "#f82f00", "#fb3e00", "#fd4900", "#fe5600", "#fe6200", "#ff6b00",
  "#ff7600", "#ff8000", "#ff8a00", "#ff9100", "#ff9a00", "#ffa301", "#ffaa01", "#ffb202", "#ffbb03", "#ffc104",
  "#ffc905", "#ffd107", "#ffd909", "#ffdf0b", "#ffe70f", "#ffef17", "#fff52b", "#fffb62", "#fffebb", "#ffffff",
];

/** Magma (matplotlib) from 14 percent up, three paler stops above: violet, magenta, salmon, pale cream. Softer than Inferno. */
export const MAGMA_STOPS = [
  "#21114e", "#29115a", "#311165", "#390f6e", "#400f74", "#491078", "#51127c", "#59157e", "#601880", "#671b80",
  "#6e1e81", "#762181", "#7e2482", "#862781", "#8c2981", "#942c80", "#9c2e7f", "#a5317e", "#ab337c", "#b3367a",
  "#bc3978", "#c43c75", "#cc3f71", "#d2426f", "#d9466b", "#e04c67", "#e75263", "#ec5860", "#f05f5e", "#f4675c",
  "#f7705c", "#f9795d", "#fa815f", "#fc8a62", "#fd9467", "#fe9d6c", "#fea772", "#feae77", "#feb77e", "#fec185",
  "#feca8d", "#fed395", "#fdda9c", "#fde3a5", "#fcecae", "#fcf6b8", "#fcfdbf", "#fcfdc8", "#fdfed0", "#fefed9",
];

export type GroundRamp = "viridis" | "inferno" | "magma" | "solar" | "fire";
export const GROUND_RAMPS: { id: GroundRamp; label: string; stops: readonly string[] }[] = [
  { id: "viridis", label: "Viridis", stops: VIRIDIS_STOPS },
  { id: "inferno", label: "Inferno", stops: INFERNO_STOPS },
  { id: "magma", label: "Magma", stops: MAGMA_STOPS },
  { id: "solar", label: "Solar", stops: SOLAR_STOPS },
  { id: "fire", label: "Fire", stops: FIRE_STOPS },
];
export const rampStops = (id: GroundRamp): readonly string[] => GROUND_RAMPS.find((r) => r.id === id)!.stops;

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/** The stops interpolated into a 256 x 1 RGBA stripe. */
export function rampStripe(stops: readonly string[] = VIRIDIS_STOPS): Uint8Array {
  const rgb = stops.map(hexToRgb);
  const out = new Uint8Array(256 * 4);
  const segs = rgb.length - 1;
  for (let x = 0; x < 256; x++) {
    const t = (x / 255) * segs;
    const i = Math.min(Math.floor(t), segs - 1);
    const f = t - i;
    const a = rgb[i]!;
    const b = rgb[i + 1]!;
    const o = x * 4;
    out[o] = Math.round(a[0] + (b[0] - a[0]) * f);
    out[o + 1] = Math.round(a[1] + (b[1] - a[1]) * f);
    out[o + 2] = Math.round(a[2] + (b[2] - a[2]) * f);
    out[o + 3] = 255;
  }
  return out;
}

export function rampTexture(device: Device, stops: readonly string[] = VIRIDIS_STOPS): Texture {
  return device.createTexture({
    format: "rgba8unorm",
    width: 256,
    height: 1,
    mipLevels: 1,
    data: rampStripe(stops),
    sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
}

/** The same ramp as a CSS gradient, for the panel strip. */
export function rampGradient(stops: readonly string[] = VIRIDIS_STOPS): string {
  return `linear-gradient(to right, ${stops.join(", ")})`;
}
