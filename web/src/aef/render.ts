import type { RenderTileResult } from "@developmentseed/deck.gl-raster";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";
import { HOLD_SCALE } from "./fuel.js";

/** What the land cover layer draws. */
export type FuelMode = "lit" | "cover";

export type FuelLightProps = {
  classTex: Texture;
  ffwiTex: Texture;
  lutTex: Texture;
  rampTex: Texture;
  /** 0 = lit by the hour's FFWI, 1 = land cover classes. */
  mode: number;
  layerIndex: number;
  hideBelow: number;
  rescaleMax: number;
  /** How many hours back the afterglow looks (0 = this hour only). */
  glowHours: number;
  /** Grain: 0 = a flat fill; above 0 the fill breaks into that many cells across the tile, each lit by the value. */
  grain: number;
};

const MODULE_NAME = "fuelLight";

/**
 * Per fragment: the pixel's class (r8, nearest) -> its column in the lookup texture (row 0 rgb the
 * class color, a its fuel weight; row 1 r its hold hours) -> either the class color, or the lit
 * value: the hour's FFWI at this spot (the weather texture, linear) times the fuel weight, and with
 * the afterglow on, the brightest of the hours behind it, each faded by exp(-age / hold) so fine
 * fuels let go of a windy hour in a few hours and timber holds it for a day. Hidden below the hide
 * line and ramped (viridis) from there to the ramp top, fading in over the first part of the ramp
 * so the ground comes up as the weather worsens instead of switching on. With grain on, the fill
 * breaks into ground-fixed grain whose density follows the value, so the ramp reads as texture.
 */
export const FuelLight = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float mode;
  float layerIndex;
  float hideBelow;
  float rescaleMax;
  float glowHours;
  float grain;
} ${MODULE_NAME};
`,
  inject: {
    "fs:#decl": `
precision highp sampler2DArray;
uniform sampler2D classTex;
uniform sampler2DArray ffwiTex;
uniform sampler2D lutTex;
uniform sampler2D rampTex;

float fuelHash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      float ci = texture(classTex, geometry.uv).r * 255.0;
      if (ci > 254.5) discard;
      vec4 lut = texelFetch(lutTex, ivec2(int(ci + 0.5), 0), 0);
      if (${MODULE_NAME}.mode > 0.5) {
        color = vec4(lut.rgb, 1.0);
      } else {
        float f = texture(ffwiTex, vec3(geometry.uv, ${MODULE_NAME}.layerIndex)).r;
        float v = isnan(f) ? -1.0 : f * lut.a;
        float hold = texelFetch(lutTex, ivec2(int(ci + 0.5), 1), 0).r * 255.0 / ${HOLD_SCALE}.0;
        int back = int(min(${MODULE_NAME}.glowHours, 48.0));
        if (hold > 0.0) {
          for (int k = 1; k <= 48; k++) {
            if (k > back) break;
            float l = ${MODULE_NAME}.layerIndex - float(k);
            if (l < 0.0) break;
            float g = texture(ffwiTex, vec3(geometry.uv, l)).r;
            if (isnan(g)) continue;
            v = max(v, g * lut.a * exp(-float(k) / hold));
          }
        }
        if (v < ${MODULE_NAME}.hideBelow) discard;
        float t = clamp((v - ${MODULE_NAME}.hideBelow) / max(${MODULE_NAME}.rescaleMax - ${MODULE_NAME}.hideBelow, 1.0), 0.0, 1.0);
        float a = min(1.0, t / 0.15);
        if (${MODULE_NAME}.grain > 0.0) {
          // grain fixed to the tile: each cell lit when its hash falls under the value, so density follows the value
          vec2 cell = floor(geometry.uv * ${MODULE_NAME}.grain);
          float h = fuelHash(cell + vec2(0.37, 0.91));
          float keep = smoothstep(h - 0.08, h + 0.08, 0.1 + 0.9 * t);
          a *= keep;
          if (a < 0.01) discard;
        }
        color = vec4(texture(rampTex, vec2(t, 0.5)).rgb, a);
      }
    `,
  },
  uniformTypes: { mode: "f32", layerIndex: "f32", hideBelow: "f32", rescaleMax: "f32", glowHours: "f32", grain: "f32" },
  getUniforms: (p: Partial<FuelLightProps>) => ({
    mode: p.mode ?? 0,
    layerIndex: p.layerIndex ?? 0,
    hideBelow: p.hideBelow ?? 0,
    rescaleMax: p.rescaleMax ?? 100,
    glowHours: p.glowHours ?? 0,
    grain: p.grain ?? 0,
    classTex: p.classTex,
    ffwiTex: p.ffwiTex,
    lutTex: p.lutTex,
    rampTex: p.rampTex,
  }),
} as const satisfies ShaderModule<FuelLightProps>;

export function fuelPipeline(p: FuelLightProps): RenderTileResult {
  return { renderPipeline: [{ module: FuelLight, props: p }] };
}
