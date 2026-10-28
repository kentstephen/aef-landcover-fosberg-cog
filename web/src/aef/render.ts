import type { RenderTileResult } from "@developmentseed/deck.gl-raster";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

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
};

const MODULE_NAME = "fuelLight";

/**
 * Per fragment: the pixel's class (r8, nearest) -> its row in the lookup texture (rgb the class
 * color, a its fuel weight) -> either the class color, or the hour's FFWI at this spot (the weather
 * texture, linear) times the fuel weight, hidden below the hide line and ramped (viridis) from
 * there to the ramp top, fading in over the first part of the ramp so the ground comes up as the
 * weather worsens instead of switching on.
 */
export const FuelLight = {
  name: MODULE_NAME,
  fs: `\
uniform ${MODULE_NAME}Uniforms {
  float mode;
  float layerIndex;
  float hideBelow;
  float rescaleMax;
} ${MODULE_NAME};
`,
  inject: {
    "fs:#decl": `
precision highp sampler2DArray;
uniform sampler2D classTex;
uniform sampler2DArray ffwiTex;
uniform sampler2D lutTex;
uniform sampler2D rampTex;
`,
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      float ci = texture(classTex, geometry.uv).r * 255.0;
      if (ci > 254.5) discard;
      vec4 lut = texelFetch(lutTex, ivec2(int(ci + 0.5), 0), 0);
      if (${MODULE_NAME}.mode > 0.5) {
        color = vec4(lut.rgb, 1.0);
      } else {
        float f = texture(ffwiTex, vec3(geometry.uv, ${MODULE_NAME}.layerIndex)).r;
        if (isnan(f)) discard;
        float v = f * lut.a;
        if (v < ${MODULE_NAME}.hideBelow) discard;
        float t = clamp((v - ${MODULE_NAME}.hideBelow) / max(${MODULE_NAME}.rescaleMax - ${MODULE_NAME}.hideBelow, 1.0), 0.0, 1.0);
        float a = min(1.0, t / 0.15);
        color = vec4(texture(rampTex, vec2(t, 0.5)).rgb, a);
      }
    `,
  },
  uniformTypes: { mode: "f32", layerIndex: "f32", hideBelow: "f32", rescaleMax: "f32" },
  getUniforms: (p: Partial<FuelLightProps>) => ({
    mode: p.mode ?? 0,
    layerIndex: p.layerIndex ?? 0,
    hideBelow: p.hideBelow ?? 0,
    rescaleMax: p.rescaleMax ?? 100,
    classTex: p.classTex,
    ffwiTex: p.ffwiTex,
    lutTex: p.lutTex,
    rampTex: p.rampTex,
  }),
} as const satisfies ShaderModule<FuelLightProps>;

export function fuelPipeline(p: FuelLightProps): RenderTileResult {
  return { renderPipeline: [{ module: FuelLight, props: p }] };
}
