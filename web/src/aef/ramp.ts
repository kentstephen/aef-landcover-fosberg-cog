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
