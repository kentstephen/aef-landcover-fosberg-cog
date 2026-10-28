import { SourceCache, SourceChunk } from "@chunkd/middleware";
import { SourceView } from "@chunkd/source";
import { SourceHttp } from "@chunkd/source-http";
import { COGLayer, type GetTileDataOptions } from "@developmentseed/deck.gl-geotiff";
import { GeoTIFF, type Overview } from "@developmentseed/geotiff";
import type { RasterModule, RenderTileResult } from "@developmentseed/deck.gl-raster";
import { CreateTexture } from "@developmentseed/deck.gl-raster/gpu-modules";
import { epsgResolver } from "@developmentseed/proj";
import type { Texture } from "@luma.gl/core";
import type { ShaderModule } from "@luma.gl/shadertools";

/**
 * Meta / WRI global canopy height v2 (DINOv3, ~1.2 m native), read from the
 * tge-labs CORS mirror on source.coop. This is the single global overview
 * COG (611 m at the base, overviews to 78 km): context under the lines, not
 * the subject. uint8 meters; 0 is both "no tree" and "no data".
 */
export const CANOPY_OVERVIEW_URL = "https://data.source.coop/tge-labs/meta-chm-v2/overview/chm_overview_z8.tif";

/** Upfront HEAD so the HTTP source knows its size before any range read (chunkd quirk). */
export async function loadCanopyGeoTIFF(href = CANOPY_OVERVIEW_URL): Promise<GeoTIFF> {
  const source = new SourceHttp(href, {});
  await source.head();
  const view = new SourceView(source, [new SourceChunk({ size: 32 * 1024 }), new SourceCache({ size: 1024 * 1024 })]);
  return await GeoTIFF.open({ dataSource: source, headerSource: view });
}

export type CanopyTileData = { width: number; height: number; texture: Texture };

async function getTileData(image: GeoTIFF | Overview, o: GetTileDataOptions): Promise<CanopyTileData> {
  const tile = await image.fetchTile(o.x, o.y, { signal: o.signal, boundless: false });
  const { width, height } = tile.array;
  const arr = tile.array;
  const src = arr.layout === "band-separate" ? arr.bands[0] : arr.data;
  if (!(src instanceof Uint8Array) || src.length < width * height) {
    throw new Error(`canopy tile: expected ${width * height} uint8, got ${src?.constructor?.name} ${src?.length}`);
  }
  const texture = o.device.createTexture({
    data: src.subarray(0, width * height),
    format: "r8unorm",
    width,
    height,
    sampler: { minFilter: "linear", magFilter: "linear", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge" },
  });
  return { width, height, texture };
}

/**
 * Height (meters, from color.r x 255) -> one green luminance ramp. Below
 * `minHeight` the pixel is discarded so the basemap shows through; the ramp
 * runs from a dark green at `minHeight` to a bright green at `maxHeight`.
 * Green is deliberate: it is the one hue the lines never use and it reads
 * as "trees" without a legend.
 */
const GreenRamp = {
  name: "canopyGreen",
  fs: `\
uniform canopyGreenUniforms {
  float minHeight;
  float maxHeight;
} canopyGreen;
`,
  inject: {
    "fs:DECKGL_FILTER_COLOR": /* glsl */ `
      float h = color.r * 255.0;
      if (h < canopyGreen.minHeight) discard;
      float t = clamp((h - canopyGreen.minHeight) / max(canopyGreen.maxHeight - canopyGreen.minHeight, 1.0), 0.0, 1.0);
      t = sqrt(t);
      color = vec4(mix(vec3(0.03, 0.16, 0.07), vec3(0.42, 0.86, 0.40), t), 1.0);
    `,
  },
  uniformTypes: { minHeight: "f32", maxHeight: "f32" },
  getUniforms: (p: { minHeight?: number; maxHeight?: number }) => ({ minHeight: p.minHeight ?? 3, maxHeight: p.maxHeight ?? 30 }),
} as const satisfies ShaderModule<{ minHeight: number; maxHeight: number }>;

export type CanopyLayerArgs = {
  geotiff: GeoTIFF;
  opacity: number;
  /** Meters; pixels below are not drawn. */
  minHeight: number;
  /** Meters at the bright end of the ramp. */
  maxHeight: number;
  beforeId?: string;
};

export function buildCanopyLayer(a: CanopyLayerArgs) {
  const renderTile = (d: CanopyTileData): RenderTileResult => {
    const renderPipeline: RasterModule[] = [
      { module: CreateTexture, props: { textureName: d.texture } },
      { module: GreenRamp, props: { minHeight: a.minHeight, maxHeight: a.maxHeight } },
    ];
    return { renderPipeline };
  };
  return new COGLayer<CanopyTileData>({
    id: "canopy-meta-chm",
    geotiff: a.geotiff,
    epsgResolver,
    getTileData,
    renderTile,
    refinementStrategy: "best-available",
    maxRequests: 12,
    opacity: a.opacity,
    updateTriggers: { renderTile: [a.minHeight, a.maxHeight] },
    beforeId: a.beforeId,
  } as never);
}
