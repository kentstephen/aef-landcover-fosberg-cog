import {
  COLORMAP_INDEX,
  createColormapTexture,
  decodeColormapSprite,
} from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import type { Device, Texture } from "@luma.gl/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapRef } from "react-map-gl/maplibre";
import { Map as MaplibreMap } from "react-map-gl/maplibre";
import type * as zarr from "zarrita";
import { buildCountyLayer } from "./counties/layer.js";
import { countyFrame, countyPeak, countyPotentialMean, loadCounties, type CountySet } from "./counties/load.js";
import type { WindMode } from "./ffwi/compute.js";
import { makeGetTileData, type FfwiTileData } from "./hrrr/get-tile-data.js";
import { dateFromInitTimeIdx, HRRR_GEOZARR_ATTRS, HRRR_LEAD_TIME_COUNT, lonLatToHrrrPixel } from "./hrrr/metadata.js";
import { makeRenderTile } from "./hrrr/render-tile.js";
import { buildSelection } from "./hrrr/selection.js";
import { SideChannel } from "./hrrr/side-channel.js";
import { openHrrr, type HrrrSource } from "./hrrr/store.js";
import { FfwiTileCache } from "./hrrr/tile-cache.js";
import { RASTER_VARIABLES, type RasterVariable } from "./hrrr/variables.js";
import { buildLineLayer, lineFfwiAt, lineRun, milesAbove } from "./lines/layer.js";
import { buildCanopyLayer, loadCanopyGeoTIFF } from "./canopy/layer.js";
import type { GeoTIFF } from "@developmentseed/geotiff";
import { KM_TO_MI, loadLines, VOLT_CLASS_LABEL, type LineSet } from "./lines/load.js";
import { DeckGlOverlay } from "./lib/deckgl-overlay.js";
import {
  fetchPotential,
  lonLatToMercator,
  POTENTIAL_LAYERS,
  samplePotential,
  type PotentialLayerId,
  type PotentialRaster,
} from "./potential/service.js";
import { ControlPanel, loadingSummary, type ColormapChoiceId, type CountyField, type CountyStat, type LineHeadline, type PeakMode, type Status, type ViewMode } from "./ui/ControlPanel.js";

const MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
/** First basemap layer above water in CARTO styles; data goes under roads/labels. */
const BEFORE_ID = "aeroway-runway";
const INITIAL_FRAME_MS = 300;
/** One CONUS-wide export of the potential raster, sampled per county once. */
const CONUS_MERC: [number, number, number, number] = [
  ...lonLatToMercator(-125.5, 24.0),
  ...lonLatToMercator(-66.0, 49.8),
] as [number, number, number, number];

export default function App() {
  const mapRef = useRef<MapRef>(null);
  const cacheRef = useRef(new FfwiTileCache());
  const [cacheVersion, setCacheVersion] = useState(0);
  useEffect(() => {
    const cache = cacheRef.current;
    const l = () => setCacheVersion(cache.version);
    cache.listeners.add(l);
    return () => {
      cache.listeners.delete(l);
    };
  }, []);

  // ---- HRRR store
  const [source, setSource] = useState<HrrrSource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [initTimeIdx, setInitTimeIdx] = useState(0);
  useEffect(() => {
    let cancelled = false;
    openHrrr()
      .then((s) => {
        if (cancelled) return;
        setSource(s);
        const latest = s.initTimeCount - 1;
        setInitTimeIdx(latest);
        // Open on now: the lead hour closest to the current time within the latest run.
        const hours = Math.round((Date.now() - dateFromInitTimeIdx(latest).getTime()) / 3_600_000);
        setLeadTimeIdx(Math.max(0, Math.min(HRRR_LEAD_TIME_COUNT - 1, hours)));
      })
      .catch((e) => setError(`HRRR store: ${String(e)}`));
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- the potential raster (once, CONUS-wide): county BP and the tooltip
  const [potentialId, setPotentialId] = useState<PotentialLayerId>("bp");
  const [potential, setPotential] = useState<PotentialRaster | null>(null);
  const [potentialLoading, setPotentialLoading] = useState(false);
  useEffect(() => {
    const ctrl = new AbortController();
    setPotentialLoading(true);
    fetchPotential(potentialId, CONUS_MERC, 1800, 1000, ctrl.signal)
      .then((r) => {
        if (!ctrl.signal.aborted) setPotential(r);
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(`Potential layer: ${String(e)}`);
      })
      .finally(() => {
        if (!ctrl.signal.aborted) setPotentialLoading(false);
      });
    return () => ctrl.abort();
  }, [potentialId]);
  useEffect(() => {
    setPotentialNorm(potentialId === "bp" ? 0.02 : 5000);
  }, [potentialId]);

  // ---- animation
  const [leadTimeIdx, setLeadTimeIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [frameDurationMs, setFrameDurationMs] = useState(INITIAL_FRAME_MS);
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      if (now - last >= frameDurationMs) {
        setLeadTimeIdx((i) => (i + 1) % HRRR_LEAD_TIME_COUNT);
        last = now;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, frameDurationMs]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code !== "Space" && e.key !== " ") return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON" || tag === "SELECT" || el?.isContentEditable) return;
      e.preventDefault();
      setIsPlaying((p) => !p);
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  }, []);

  // ---- styling
  const [windMode, setWindMode] = useState<WindMode>("sustained");
  const [colormapId, setColormapId] = useState<ColormapChoiceId>("solar");
  /** FFWI ramp top (the county view and the FFWI raster share it). */
  const [rescaleMax, setRescaleMax] = useState(100);
  /** FFWI below this is hidden on the raster and is the county "share" threshold. */
  const [hideBelow, setHideBelow] = useState(20);
  // The lines are the subject; the field under them (counties or raster) is off until asked for.
  const [showRaster, setShowRaster] = useState(false);
  const [showCounties, setShowCounties] = useState(false);
  const [rasterOpacity, setRasterOpacity] = useState(0.7);
  /** County BP ramp top (display units). */
  const [potentialNorm, setPotentialNorm] = useState(0.02);
  /** Fuel-moisture timelag in hours; 0 = Fosberg as published. Applied in the cache (counties, tooltip) and the shader (raster). */
  const [tauHours, setTauHours] = useState(0);
  useEffect(() => {
    cacheRef.current.setTau(tauHours);
  }, [tauHours]);
  /** Trailing window for the county "hours above the line" stat. */
  const [windowHours, setWindowHours] = useState(12);
  /** county (default) = one FFWI number per county; raster = the 3 km field of FFWI or one of its inputs. */
  const [viewMode, setViewMode] = useState<ViewMode>("county");
  const [rasterVariable, setRasterVariable] = useState<RasterVariable>("ffwi");
  const countyView = viewMode === "county";
  const varDef = RASTER_VARIABLES[rasterVariable];

  // ---- county view
  const [countyField, setCountyField] = useState<CountyField>("ffwi");
  const [countyStat, setCountyStat] = useState<CountyStat>("share");
  const [peakMode, setPeakMode] = useState<PeakMode>("peak");
  // Each raster variable brings its own colormap (temperature: cool-warm); the county view is solar,
  // except the peak-hour map, which is a time axis and gets viridis.
  useEffect(() => {
    setColormapId(countyView ? (countyField === "peak" ? "viridis" : "solar") : varDef.defaultColormap);
  }, [countyView, countyField, varDef]);
  const [counties, setCounties] = useState<CountySet | null>(null);
  useEffect(() => {
    if (!countyView || counties) return;
    const ctrl = new AbortController();
    loadCounties(ctrl.signal)
      .then((c) => {
        if (!ctrl.signal.aborted) setCounties(c);
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(`Counties: ${String(e)}`);
      });
    return () => ctrl.abort();
  }, [countyView, counties]);
  const countyBp = useMemo(() => (counties && potential ? countyPotentialMean(counties, potential) : null), [counties, potential]);
  /** Shards that hold at least one county pixel (21 of the grid's 24: the rest are open water or outside the US). */
  const countyShardKeys = useMemo(() => {
    if (!counties) return null;
    const keys = new Set<string>();
    for (let j = 0; j < counties.pxX.length; j++) keys.add(FfwiTileCache.keyFor(counties.pxX[j]!, counties.pxY[j]!));
    return [...keys];
  }, [counties]);
  const [countyHover, setCountyHover] = useState<{ i: number; x: number; y: number } | null>(null);

  // ---- transmission lines (HIFLD, preprocessed to lines.bin): the subject; the weather lights them.
  const [showLines, setShowLines] = useState(true);
  const [lineGhost, setLineGhost] = useState(0.18);
  /** Width at the ramp top, px (1 px at the hide line). */
  const [lineMaxWidth, setLineMaxWidth] = useState(3);
  const [lines, setLines] = useState<LineSet | null>(null);
  // ---- canopy height (Meta / WRI, source.coop overview COG): context under the lines, static.
  const [showCanopy, setShowCanopy] = useState(true);
  const [canopyOpacity, setCanopyOpacity] = useState(0.3);
  const [canopyMinHeight, setCanopyMinHeight] = useState(3);
  const [canopyMaxHeight, setCanopyMaxHeight] = useState(30);
  const [canopy, setCanopy] = useState<GeoTIFF | null>(null);
  useEffect(() => {
    if (!showCanopy || canopy) return;
    let gone = false;
    loadCanopyGeoTIFF()
      .then((g) => { if (!gone) setCanopy(g); })
      .catch((e) => { if (!gone) setError(`Canopy: ${String(e)}`); });
    return () => { gone = true; };
  }, [showCanopy, canopy]);
  const [lineHover, setLineHover] = useState<{ i: number; x: number; y: number } | null>(null);
  useEffect(() => {
    if (!showLines || lines) return;
    const ctrl = new AbortController();
    loadLines(ctrl.signal)
      .then((l) => {
        if (!ctrl.signal.aborted) setLines(l);
      })
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(`Lines: ${String(e)}`);
      });
    return () => ctrl.abort();
  }, [showLines, lines]);

  const [device, setDevice] = useState<Device | null>(null);
  const [colormapImage, setColormapImage] = useState<ImageData | null>(null);
  const [colormapTexture, setColormapTexture] = useState<Texture | null>(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const resp = await fetch(colormapsPngUrl);
      const image = await decodeColormapSprite(await resp.arrayBuffer());
      if (!cancelled) setColormapImage(image);
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (!device || !colormapImage) return;
    setColormapTexture(createColormapTexture(device, colormapImage));
  }, [device, colormapImage]);
  const colormapIndex = COLORMAP_INDEX[colormapId];

  // New run or wind mode => tiles refetch (layer id changes); drop the cache.
  useEffect(() => {
    cacheRef.current.clear();
  }, [initTimeIdx, windMode]);

  // In the county view nothing else fills the FFWI cache: the side channel
  // fetches every shard the counties touch (all 24). It skips cached shards.
  useEffect(() => {
    if (!countyView || !source || !counties) return;
    const sc = new SideChannel(source.arrays, cacheRef.current);
    void sc.ensure(counties.pxX, counties.pxY, initTimeIdx, windMode);
    return () => sc.abort();
  }, [countyView, counties, source, initTimeIdx, windMode]);
  // The lines need every shard they touch lit, in either view. In the raster view the ZarrLayer
  // only fetches the viewport's tiles, so the side channel fills the rest (it skips cached shards;
  // a short delay lets the raster tiles go first, as in road-risk).
  useEffect(() => {
    if (!showLines || !source || !lines) return;
    const sc = new SideChannel(source.arrays, cacheRef.current);
    const timer = setTimeout(() => void sc.ensure(lines.px, lines.py, initTimeIdx, windMode), countyView ? 0 : 1200);
    return () => {
      clearTimeout(timer);
      sc.abort();
    };
  }, [showLines, lines, source, initTimeIdx, windMode, countyView]);
  // FFWI for every pixel the lines touch, all 49 leads, once per cache change (tau included);
  // frames and the headline are lookups into it.
  const lineRunData = useMemo(
    () => (showLines && lines ? lineRun(lines, cacheRef.current) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [showLines, lines, cacheVersion],
  );
  const lineMiles = useMemo(
    () => (lines && lineRunData ? milesAbove(lines, lineRunData, hideBelow) : null),
    [lines, lineRunData, hideBelow],
  );
  const lineFfwiNow = useMemo(
    () => (lines && lineRunData ? lineFfwiAt(lines, lineRunData, leadTimeIdx) : null),
    [lines, lineRunData, leadTimeIdx],
  );
  const lineHeadline: LineHeadline | null = useMemo(() => {
    if (!lines || !lineRunData || !lineMiles) return null;
    let peakLead = 0;
    for (let l = 1; l < lineMiles.length; l++) if (lineMiles[l]! > lineMiles[peakLead]!) peakLead = l;
    return {
      milesNow: lineMiles[leadTimeIdx]!,
      milesTotal: lines.totalKm * KM_TO_MI,
      milesLoaded: lineRunData.milesLoaded,
      peakLead,
      milesPeak: lineMiles[peakLead]!,
    };
  }, [lines, lineRunData, lineMiles, leadTimeIdx]);

  // Whole-run pass per county (mean and share per lead, peak, onset) for the peak map and the decay
  // stat; recomputed as shards arrive and when tau changes.
  const needRun = countyView && showCounties && countyField === "peak";
  const countyPeakData = useMemo(
    () => (needRun && counties ? countyPeak(counties, cacheRef.current, hideBelow) : null),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [needRun, counties, hideBelow, cacheVersion],
  );
  const countyFrameData = useMemo(
    () => (countyView && showCounties && counties ? countyFrame(counties, cacheRef.current, leadTimeIdx, hideBelow, windowHours) : null),
    // cacheVersion: new shards arrive while the lead is unchanged, and setTau bumps it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [countyView, showCounties, counties, leadTimeIdx, hideBelow, windowHours, cacheVersion],
  );

  // ---- hover (raster view): map mousemove -> HRRR pixel -> FFWI cache + BP export.
  const [hover, setHover] = useState<{ lon: number; lat: number; x: number; y: number } | null>(null);
  const hoverInfo = useMemo(() => {
    if (!hover || countyView) return null;
    const px = lonLatToHrrrPixel(hover.lon, hover.lat);
    const wx = px ? cacheRef.current.sample(px.x, px.y, leadTimeIdx) : null;
    const bp = potential ? samplePotential(potential, hover.lon, hover.lat) : NaN;
    const ffwi = wx?.ffwi ?? NaN;
    // Speak where the map is lit: FFWI above the hide line, or any loaded pixel for the other variables.
    const lit = rasterVariable === "ffwi" ? Number.isFinite(ffwi) && ffwi >= hideBelow : wx !== null;
    return { wx, bp, lit };
  }, [hover, countyView, leadTimeIdx, cacheVersion, potential, rasterVariable, hideBelow]);

  // ---- layers
  const selection = useMemo(() => buildSelection(initTimeIdx), [initTimeIdx]);
  const getTileData = useMemo(
    () => (source ? makeGetTileData({ arrays: source.arrays, cache: cacheRef.current, windMode, variable: rasterVariable }) : null),
    [source, windMode, rasterVariable],
  );
  const renderTile = useCallback(
    (data: FfwiTileData) => {
      if (!colormapTexture) return { renderPipeline: [] };
      const isFfwi = rasterVariable === "ffwi";
      return makeRenderTile({
        layerIndex: leadTimeIdx,
        colormapTexture,
        colormapIndex,
        reversed: varDef.reversed,
        hideBelow: isFfwi ? hideBelow : Number.NEGATIVE_INFINITY,
        rescaleMin: varDef.min,
        rescaleMax: isFfwi ? rescaleMax : varDef.max,
        tauHours,
      })(data);
    },
    [leadTimeIdx, colormapTexture, colormapIndex, hideBelow, rescaleMax, rasterVariable, varDef, tauHours],
  );
  const pot = POTENTIAL_LAYERS[potentialId];

  const deckLayers = useMemo(() => {
    const out: unknown[] = [];
    if (showCanopy && canopy) {
      out.push(buildCanopyLayer({ geotiff: canopy, opacity: canopyOpacity, minHeight: canopyMinHeight, maxHeight: canopyMaxHeight, beforeId: BEFORE_ID }));
    }
    if (countyView && showCounties && counties && countyFrameData && countyBp && colormapImage) {
      out.push(
        buildCountyLayer({
          counties,
          frame: countyFrameData,
          bpMean: countyBp,
          peak: countyPeakData,
          peakMode,
          field: countyField,
          stat: countyStat,
          colormap: colormapImage,
          colormapIndex,
          hideBelow,
          ffwiMax: rescaleMax,
          bpFull: potentialNorm,
          windowHours,
          opacity: rasterOpacity,
          updateKey: [leadTimeIdx, cacheVersion, countyField, countyStat, colormapIndex, hideBelow, rescaleMax, potentialNorm, countyBp, windowHours],
          onHover: (i, x, y) => setCountyHover(i === null ? null : { i, x, y }),
          onClick: () => {},
          beforeId: BEFORE_ID,
        }),
      );
    }
    if (!countyView && showRaster && source && getTileData && colormapTexture) {
      out.push(
        new ZarrLayer<zarr.Readable, "float32", FfwiTileData>({
          id: `hrrr-zarr-${initTimeIdx}-${windMode}-${rasterVariable}`,
          node: source.arrays.temperature_2m,
          metadata: HRRR_GEOZARR_ATTRS,
          selection,
          getTileData,
          renderTile,
          maxRequests: 20,
          maxCacheSize: 12,
          opacity: rasterOpacity,
          updateTriggers: { renderTile: [leadTimeIdx, colormapIndex, hideBelow, rescaleMax, rasterVariable, tauHours] },
          beforeId: BEFORE_ID,
        } as never),
      );
    }
    if (showLines && lines && lineFfwiNow && colormapImage) {
      out.push(
        buildLineLayer({
          lines,
          ffwi: lineFfwiNow,
          hideBelow,
          ffwiMax: rescaleMax,
          colormap: colormapImage,
          colormapIndex,
          ghostAlpha: lineGhost,
          maxWidth: lineMaxWidth,
          onHover: (i, x, y) => setLineHover(i === null ? null : { i, x, y }),
          beforeId: BEFORE_ID,
        }),
      );
    }
    return out;
  }, [
    showCanopy, canopy, canopyOpacity, canopyMinHeight, canopyMaxHeight,
    showLines, lines, lineFfwiNow, lineGhost, lineMaxWidth, showCounties,
    countyView, counties, countyFrameData, countyBp, colormapImage, countyField, countyStat, colormapIndex, hideBelow, countyPeakData, peakMode,
    rescaleMax, potentialNorm, rasterOpacity, leadTimeIdx, cacheVersion, showRaster, source, getTileData, colormapTexture,
    initTimeIdx, windMode, rasterVariable, selection, renderTile, windowHours,
  ]);

  // The raster tiles the whole grid (24 shards); the county view and the lines only need the shards they touch.
  const lineShardKeys = useMemo(() => {
    if (!lines) return null;
    const keys = new Set<string>();
    for (let j = 0; j < lines.n; j++) keys.add(FfwiTileCache.keyFor(lines.px[j]!, lines.py[j]!));
    return [...keys];
  }, [lines]);
  const neededShardKeys = useMemo(() => {
    if (!countyView && !showLines) return null;
    const keys = new Set<string>();
    if (countyView && countyShardKeys) for (const k of countyShardKeys) keys.add(k);
    if (showLines && lineShardKeys) for (const k of lineShardKeys) keys.add(k);
    return keys.size ? [...keys] : null;
  }, [countyView, showLines, countyShardKeys, lineShardKeys]);
  const tilesTotal = neededShardKeys ? neededShardKeys.length : 24;
  const tilesLoaded = neededShardKeys
    ? neededShardKeys.filter((k) => cacheRef.current.has(k)).length
    : cacheRef.current.size;
  const status: Status = {
    backend: source?.backend ?? null,
    tilesLoaded,
    tilesTotal,
    potentialReady: potential !== null,
    potentialLoading,
    countiesNeeded: countyView,
    countiesReady: counties !== null,
    linesNeeded: showLines,
    linesReady: lines !== null,
    error,
  };
  const loading = loadingSummary(status);

  const tooltipStyle = (x: number, y: number): React.CSSProperties => ({
    position: "absolute",
    left: x + 14,
    top: y + 14,
    pointerEvents: "none",
    background: "rgba(20,20,24,0.92)",
    color: "#eee",
    fontSize: 12,
    lineHeight: 1.4,
    padding: "6px 9px",
    borderRadius: 6,
    zIndex: 2000,
    whiteSpace: "nowrap",
  });

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <MaplibreMap
        ref={mapRef}
        initialViewState={{ longitude: -98, latitude: 38.5, zoom: 4 }}
        mapStyle={MAP_STYLE_URL}
        attributionControl={false}
        onMouseMove={(e) => setHover({ lon: e.lngLat.lng, lat: e.lngLat.lat, x: e.point.x, y: e.point.y })}
        onMouseOut={() => setHover(null)}
      >
        <DeckGlOverlay layers={deckLayers as never} interleaved onDeviceInitialized={setDevice} />
      </MaplibreMap>
      {loading ? (
        <div
          style={{
            position: "absolute",
            top: 14,
            left: "50%",
            transform: "translateX(-50%)",
            pointerEvents: "none",
            background: "rgba(20,20,24,0.85)",
            color: "#eee",
            fontSize: 12,
            padding: "6px 12px",
            borderRadius: 999,
            zIndex: 1500,
            whiteSpace: "nowrap",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span
            style={{
              width: 90,
              height: 4,
              borderRadius: 2,
              background: "rgba(255,255,255,0.18)",
              overflow: "hidden",
              display: "inline-block",
            }}
          >
            <span style={{ display: "block", height: "100%", width: `${Math.round(loading.fraction * 100)}%`, background: "#f6c343", transition: "width 300ms" }} />
          </span>
          {loading.label}
        </div>
      ) : null}
      <ControlPanel
        leadTimeIdx={leadTimeIdx}
        initTimeIdx={initTimeIdx}
        initTimeCount={source?.initTimeCount ?? 0}
        isPlaying={isPlaying}
        frameDurationMs={frameDurationMs}
        onInitTimeIdxChange={setInitTimeIdx}
        onLeadTimeIdxChange={setLeadTimeIdx}
        onPlayPauseToggle={() => setIsPlaying((p) => !p)}
        onFrameDurationMsChange={setFrameDurationMs}
        windMode={windMode}
        onWindModeChange={setWindMode}
        colormapId={colormapId}
        onColormapChange={setColormapId}
        rescaleMax={rescaleMax}
        onRescaleMaxChange={setRescaleMax}
        hideBelow={hideBelow}
        onHideBelowChange={setHideBelow}
        showRaster={showRaster}
        onShowRasterChange={setShowRaster}
        showCounties={showCounties}
        onShowCountiesChange={setShowCounties}
        rasterOpacity={rasterOpacity}
        onRasterOpacityChange={setRasterOpacity}
        potentialId={potentialId}
        onPotentialIdChange={setPotentialId}
        potentialNorm={potentialNorm}
        onPotentialNormChange={setPotentialNorm}
        viewMode={viewMode}
        onViewModeChange={setViewMode}
        rasterVariable={rasterVariable}
        onRasterVariableChange={setRasterVariable}
        countyField={countyField}
        onCountyFieldChange={setCountyField}
        countyStat={countyStat}
        onCountyStatChange={setCountyStat}
        peakMode={peakMode}
        onPeakModeChange={setPeakMode}
        countyCount={counties?.features.length ?? 0}
        tauHours={tauHours}
        onTauHoursChange={setTauHours}
        windowHours={windowHours}
        onWindowHoursChange={setWindowHours}
        showCanopy={showCanopy}
        onShowCanopyChange={setShowCanopy}
        canopyReady={canopy !== null}
        canopyOpacity={canopyOpacity}
        onCanopyOpacityChange={setCanopyOpacity}
        canopyMinHeight={canopyMinHeight}
        onCanopyMinHeightChange={setCanopyMinHeight}
        canopyMaxHeight={canopyMaxHeight}
        onCanopyMaxHeightChange={setCanopyMaxHeight}
        showLines={showLines}
        onShowLinesChange={setShowLines}
        lineGhost={lineGhost}
        onLineGhostChange={setLineGhost}
        lineMaxWidth={lineMaxWidth}
        onLineMaxWidthChange={setLineMaxWidth}
        lineHeadline={lineHeadline}
        status={status}
      />
      {lineHover && lines && lineFfwiNow ? (() => {
        const i = lineHover.i;
        const wx = cacheRef.current.sample(lines.px[i]!, lines.py[i]!, leadTimeIdx);
        return (
          <div style={tooltipStyle(lineHover.x, lineHover.y)}>
            <div style={{ fontWeight: 600 }}>Transmission line, {VOLT_CLASS_LABEL[lines.vc[i]!]}</div>
            <div>
              {wx && Number.isFinite(wx.ffwi)
                ? `FFWI ${wx.ffwi.toFixed(0)} · wind ${wx.windMph} mph · RH ${wx.rh}% · ${wx.tc} °C`
                : "forecast not loaded yet"}
              {tauHours > 0 ? ` (fuel lag ${tauHours} h)` : ""}
            </div>
          </div>
        );
      })() : null}
      {countyView && countyHover && !lineHover && counties && countyFrameData && countyBp ? (
        <div style={tooltipStyle(countyHover.x, countyHover.y)}>
          <div style={{ fontWeight: 600 }}>
            {counties.name[countyHover.i]} {counties.features[countyHover.i]!.properties.LSAD ?? "County"}, {counties.state[countyHover.i]}
          </div>
          {countyField === "peak" && countyPeakData ? (
            <div>
              {countyPeakData.peakLead[countyHover.i]! >= 0
                ? `peak mean FFWI ${countyPeakData.peakValue[countyHover.i]!.toFixed(0)} at +${countyPeakData.peakLead[countyHover.i]} h · ` +
                  (countyPeakData.onsetLead[countyHover.i]! >= 0
                    ? `first ≥ ${hideBelow} at +${countyPeakData.onsetLead[countyHover.i]} h`
                    : `never reaches ${hideBelow}`)
                : "not loaded yet"}
            </div>
          ) : null}
          <div>
            FFWI this hour:{" "}
            {Number.isFinite(countyFrameData.mean[countyHover.i]!)
              ? `mean ${countyFrameData.mean[countyHover.i]!.toFixed(0)} · max ${countyFrameData.max[countyHover.i]!.toFixed(0)} · ${Math.round(
                  100 * countyFrameData.share[countyHover.i]!,
                )}% of pixels ≥ ${hideBelow} · ${countyFrameData.hours[countyHover.i]!.toFixed(1)} of the last ${countyFrameData.windowSeen} h ≥ ${hideBelow}`
              : "not loaded yet"}
            {tauHours > 0 ? ` (fuel lag ${tauHours} h)` : ""}
          </div>
          <div>
            {pot.shortLabel} mean {Number.isFinite(countyBp[countyHover.i]!) ? pot.format(countyBp[countyHover.i]!) : "n/a"} ·{" "}
            {counties.pxStart[countyHover.i + 1]! - counties.pxStart[countyHover.i]!} HRRR pixels sampled
          </div>
        </div>
      ) : null}
      {!countyView && hover && !lineHover && hoverInfo && hoverInfo.lit && hoverInfo.wx ? (
        <div style={tooltipStyle(hover.x, hover.y)}>
          <div style={{ fontWeight: 600 }}>
            {Number.isFinite(hoverInfo.wx.ffwi) ? `FFWI ${hoverInfo.wx.ffwi.toFixed(1)}` : "FFWI n/a"}
            {tauHours > 0 && Number.isFinite(hoverInfo.wx.ffwiInstant)
              ? ` (${tauHours} h fuel lag · instant ${hoverInfo.wx.ffwiInstant.toFixed(1)})`
              : ""}
          </div>
          <div>
            wind {hoverInfo.wx.windMph} mph · RH {hoverInfo.wx.rh}% · {hoverInfo.wx.tc} °C ·{" "}
            {pot.shortLabel} {Number.isFinite(hoverInfo.bp) ? pot.format(hoverInfo.bp) : "n/a"}
          </div>
          <div style={{ color: "#aaa" }}>
            {hover.lat.toFixed(2)}, {hover.lon.toFixed(2)}
          </div>
        </div>
      ) : null}
    </div>
  );
}
