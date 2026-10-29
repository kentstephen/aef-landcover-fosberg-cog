import {
  COLORMAP_INDEX,
  createColormapTexture,
  decodeColormapSprite,
} from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { ZarrLayer } from "@developmentseed/deck.gl-zarr";
import { PathLayer } from "@deck.gl/layers";
import type { Device, Texture } from "@luma.gl/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MapRef } from "react-map-gl/maplibre";
import { Map as MaplibreMap } from "react-map-gl/maplibre";
import type * as zarr from "zarrita";
import { fuelOf } from "./aef/fuel.js";
// import { fireLayers, firesInRun, loadFires, type Fires } from "./fires.js";
// the burn perimeters (perimeters.ts): not drawn for now; TODO draw them from GeoArrow (deck.gl-geoarrow) rather
// than a 7 MB GeoJSON parsed on the main thread, and only while each fire burned
// import { loadPerimeters, type Perimeter, perimeterLayer, perimetersInRun } from "./perimeters.js";
import { HEADS, NO_CLASS, type Teacher } from "./aef/head.js";
import { AefLayers } from "./aef/layers.js";
import { type AefArray, MOSAIC_MIN_ZOOM, openAefMosaic } from "./aef/mosaic.js";
import { AefPool } from "./aef/pool.js";
import { CoarsePool } from "./aef/coarse.js";
import { type AefIndex, loadAefIndex } from "./aef/index.js";
import type { WindMode } from "./ffwi/compute.js";
import { makeGetTileData, type FfwiTileData } from "./hrrr/get-tile-data.js";
import { dateFromInitTimeIdx, HRRR_GEOZARR_ATTRS, HRRR_LEAD_TIME_COUNT, initTimeIdxFromDate, lonLatToHrrrPixel } from "./hrrr/metadata.js";
import { makeRenderTile } from "./hrrr/render-tile.js";
import { buildSelection } from "./hrrr/selection.js";
import { type ContourPath, ffwiContours } from "./hrrr/contours.js";
import { SideChannel } from "./hrrr/side-channel.js";
import { openHrrr, type HrrrSource } from "./hrrr/store.js";
import { FfwiTileCache } from "./hrrr/tile-cache.js";
import { RASTER_VARIABLES, type RasterVariable } from "./hrrr/variables.js";
import { HOURS_PER_DAY, WeekWeather, weekHourDate, type WeekSpec } from "./hrrr/week.js";
import { cacheSource } from "./aef/weather.js";
import { DeckGlOverlay } from "./lib/deckgl-overlay.js";
import { ControlPanel, loadingSummary, type ColormapChoiceId, type FuelChoice, type IgnScale, type Status, type TimeMode } from "./ui/ControlPanel.js";

const MAP_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
/** First basemap layer above water in CARTO styles; data goes under roads/labels. */
const BEFORE_ID = "aeroway-runway";
/** 100 ms a frame: a week in 17 s. */
const INITIAL_FRAME_MS = 100;
/**
 * Opens on the whole country. The single-run mode opens on the 12Z run of April 22, 2026, the second of
 * the two wide red flag days, playing from +0 h so the afternoon peak comes up over the first frames and the
 * afterglow carries it into the 23rd. (March 14, 2025 12Z is another: the Southern Plains
 * outbreak, extreme wind and single-digit humidity from Texas into the Mid-South.)
 */
const START_VIEW = { longitude: -97, latitude: 38, zoom: 4.6 };
const START_INIT = new Date("2026-04-22T12:00:00Z");
const START_LEAD = 0;
/**
 * The week the map opens on, over the whole country: April 19 to 25, 2026. On the 21st and 22nd red flag
 * warnings covered 11 to 12 million people across eight states, from east-central New Mexico up the High Plains
 * into South Dakota and across the Intermountain West, with Particularly Dangerous Situation warnings in
 * Nebraska and southern Colorado: the widest fire weather picture of the year. The ground is the 2025
 * embeddings. (Other weeks: February 15 to 21, 2026, the Ranger Road day on the 17th along a thousand-mile
 * corridor from New Mexico to Wyoming; March 10 to 16, 2026, the Morrill Fire in the Nebraska Sandhills;
 * July 29 to August 4, 2026, Washington's first PDS red flag warning on August 1.)
 */
const START_WEEK = new Date("2026-04-19T00:00:00Z");
const START_DAYS = 7;
/** Terrain under the ground: AWS Terrain Tiles (Mapzen Terrarium), public, 256 px, to zoom 15. */
const DEM_TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const HILLSHADE_ID = "aef-hillshade";

/** Relief under everything: a hillshade layer from the terrain tiles, below the first data layer. */
function addHillshade(map: import("maplibre-gl").Map, on: boolean) {
  if (!map.getSource("aef-dem")) {
    map.addSource("aef-dem", { type: "raster-dem", tiles: [DEM_TILES], encoding: "terrarium", tileSize: 256, maxzoom: 15 });
  }
  if (!map.getLayer(HILLSHADE_ID)) {
    const before = map.getLayer("aef-coarse") ? "aef-coarse" : BEFORE_ID;
    map.addLayer(
      {
        id: HILLSHADE_ID,
        type: "hillshade",
        source: "aef-dem",
        paint: {
          "hillshade-shadow-color": "#000000",
          "hillshade-highlight-color": "#9aa7b8",
          "hillshade-accent-color": "#000000",
          "hillshade-exaggeration": 0.55,
          "hillshade-illumination-direction": 335,
        },
      },
      before,
    );
  }
  map.setLayoutProperty(HILLSHADE_ID, "visibility", on ? "visible" : "none");
}

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
        setInitTimeIdx(initTimeIdxFromDate(START_INIT, s.initTimeCount - 1));
        setLeadTimeIdx(START_LEAD);
      })
      .catch((e) => setError(`HRRR store: ${String(e)}`));
    return () => {
      cancelled = true;
    };
  }, []);

  // ---- the embeddings: the tile index, the mosaic, the worker pool, the layer controller
  const poolRef = useRef<AefPool | null>(null);
  const coarseRef = useRef<CoarsePool | null>(null);
  const [aefIndex, setAefIndex] = useState<AefIndex | null>(null);
  const [mosaic, setMosaic] = useState<{ arr: AefArray; attrs: unknown } | null>(null);
  useEffect(() => {
    let gone = false;
    loadAefIndex(import.meta.env.BASE_URL)
      .then((i) => { if (!gone) setAefIndex(i); })
      .catch((e) => { if (!gone) setError(`AEF index: ${String(e)}`); });
    openAefMosaic()
      .then((m) => { if (!gone) setMosaic(m); })
      .catch((e) => { if (!gone) setError(`AEF mosaic: ${String(e)}`); });
    return () => { gone = true; };
  }, []);
  // the real ignitions as points over the ground (fires.ts, fires.bin): not drawn; the picture is the
  // ground and the weather
  // const [fires, setFires] = useState<Fires | null>(null);
  // useEffect(() => { loadFires(import.meta.env.BASE_URL).then(setFires).catch((e) => setError(`fires: ${String(e)}`)); }, []);
  // const [showFires, setShowFires] = useState(true);
  // const fireList = useMemo(() => (fires && source ? firesInRun(fires, dateFromInitTimeIdx(initTimeIdx)) : []), [fires, source, initTimeIdx]);
  // ---- the burn perimeters of the fires (the year after the embeddings), the frame for the picture
  // const [perimeters, setPerimeters] = useState<Perimeter[]>([]);
  // useEffect(() => { loadPerimeters(import.meta.env.BASE_URL, AEF_YEAR + 1).then(setPerimeters).catch((e) => setError(`perimeters: ${String(e)}`)); }, []);
  // const [showPerimeters, setShowPerimeters] = useState(true);
  // const runPerimeters = useMemo(() => (source ? perimetersInRun(perimeters, dateFromInitTimeIdx(initTimeIdx)) : []), [perimeters, source, initTimeIdx]);
  const [aefVersion, setAefVersion] = useState(0);
  const needWeatherRef = useRef<(px: Int16Array, py: Int16Array) => void>(() => {});
  const aefLayersRef = useRef<AefLayers | null>(null);
  const aefLayers = useMemo(() => {
    if (!aefIndex || !mosaic) return null;
    if (!poolRef.current) poolRef.current = new AefPool();
    if (!coarseRef.current) coarseRef.current = new CoarsePool(import.meta.env.BASE_URL);
    const l = new AefLayers(
      poolRef.current,
      coarseRef.current,
      cacheRef.current,
      aefIndex,
      mosaic,
      (px, py) => needWeatherRef.current(px, py),
      () => setAefVersion((v) => v + 1),
    );
    return l;
  }, [aefIndex, mosaic]);
  // StrictMode runs the memo twice in development and keeps the first result: the ref follows the kept one
  useEffect(() => {
    aefLayersRef.current = aefLayers;
  }, [aefLayers]);

  // ---- animation
  const timeModeRef = useRef<TimeMode>("week");
  /** Week mode: the hour after this one for the play loop (skips days not yet in hand). */
  const nextHourRef = useRef<(h: number) => number>((h) => (h + 1) % depthRef.current);
  const depthRef = useRef(START_DAYS * HOURS_PER_DAY);
  const [leadTimeIdx, setLeadTimeIdx] = useState(START_LEAD);
  const [isPlaying, setIsPlaying] = useState(true);
  const [frameDurationMs, setFrameDurationMs] = useState(INITIAL_FRAME_MS);
  useEffect(() => {
    if (!isPlaying) return;
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      if (now - last >= frameDurationMs) {
        if (timeModeRef.current === "week") setHourIdx((i) => nextHourRef.current(i));
        else setLeadTimeIdx((i) => (i + 1) % HRRR_LEAD_TIME_COUNT);
        last = now;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [isPlaying, frameDurationMs]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const space = e.code === "Space" || e.key === " ";
      const full = e.key === "f" || e.key === "F";
      if (!space && !full) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON" || tag === "SELECT" || el?.isContentEditable) return;
      e.preventDefault();
      if (space) setIsPlaying((p) => !p);
      else if (document.fullscreenElement) void document.exitFullscreen();
      else void document.documentElement.requestFullscreen();
    };
    // the browser's real full screen (F): the map takes the new size
    const onFull = () => mapRef.current?.getMap().resize();
    window.addEventListener("keydown", onKey, { capture: true });
    document.addEventListener("fullscreenchange", onFull);
    return () => {
      window.removeEventListener("keydown", onKey, { capture: true });
      document.removeEventListener("fullscreenchange", onFull);
    };
  }, []);

  // ---- styling
  const [windMode, setWindMode] = useState<WindMode>("sustained");
  const [colormapId, setColormapId] = useState<ColormapChoiceId>("solar");
  /** Ramp top for the ground and the field. */
  const [rescaleMax, setRescaleMax] = useState(100);
  /** Below this nothing is drawn (ground or field). */
  const [hideBelow, setHideBelow] = useState(20);
  const [showRasterState, setShowRasterState] = useState(false);
  const showRaster = showRasterState;
  const [rasterOpacity, setRasterOpacity] = useState(0.7);
  const [rasterVariable, setRasterVariable] = useState<RasterVariable>("ffwi");
  const varDef = RASTER_VARIABLES[rasterVariable];
  useEffect(() => {
    setColormapId(varDef.defaultColormap);
  }, [varDef]);
  /** Fuel-moisture timelag in hours; 0 = Fosberg as published. */
  const [tauHours, setTauHours] = useState(0);
  useEffect(() => {
    cacheRef.current.setTau(tauHours);
  }, [tauHours]);
  const [fuel, setFuelState] = useState<FuelChoice>("ignition");
  /** Ignition mode: hidden below this probability, ramp top at pMax. */
  const [pHide, setPHide] = useState(0.5);
  const [pMax, setPMax] = useState(0.95);
  /** How the ignition logit meets the ramp: probability, logit, or rank among the pixels in view this hour. */
  const [ignScale, setIgnScale] = useState<IgnScale>("probability");
  const [zLo, setZLo] = useState(0);
  const [zHi, setZHi] = useState(4);
  /** Rank scale: the quantiles in view that sit at the hide line and the ramp top. */
  const [qLo, setQLo] = useState(0.5);
  const [qHi, setQHi] = useState(0.99);
  /** The weather term left out: the ground alone. */
  const [useWeather, setUseWeather] = useState(true);
  /** The standing picture: the ground's probability faint all the time, opaque where the hour's weather crosses the gate. */
  const [standingAlpha, setStandingAlpha] = useState(0.25);
  const [wxGate, setWxGate] = useState(1.0);
  // one picture at a time: drawing the field puts the ground off, and the ground back on puts the field away
  const setShowRaster = useCallback((on: boolean) => {
    setShowRasterState(on);
    if (on) setFuelState("off");
  }, []);
  const setFuel = useCallback((f: FuelChoice) => {
    setFuelState(f);
    if (f !== "off") setShowRasterState(false);
  }, []);
  const [teacher, setTeacher] = useState<Teacher>("nlcd");
  const [fuelOpacity, setFuelOpacity] = useState(0.9);
  /** Hours of afterglow on the ground (0 = this hour only). */
  const [glowHours, setGlowHours] = useState(24);
  /** Grain cells across a tile (0 = flat fill). */
  const [grain, setGrain] = useState(0);
  /** The ground drawn as light (additive) rather than paint. */
  const [light, setLight] = useState(false);
  // relief (AWS Terrarium hillshade) under the ground: off, it only makes the ground harder to read
  const [relief, setRelief] = useState(false);
  /** Contours of the FFWI field over the ground (0 = none). */
  const [contourStep, setContourStep] = useState(0);
  const [zoom, setZoom] = useState(START_VIEW.zoom);
  const [view, setView] = useState<[number, number, number, number] | null>(null);
  const readView = (m: import("maplibre-gl").Map) => {
    const b = m.getBounds();
    setView([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]);
  };
  useEffect(() => {
    const m = mapRef.current?.getMap();
    if (m && m.isStyleLoaded()) addHillshade(m, relief);
  }, [relief]);

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

  // ---- time: one 48 h run, or a week of 12Z runs stitched hour by hour
  const [timeMode, setTimeMode] = useState<TimeMode>("week");
  const [weekStart, setWeekStart] = useState<Date>(START_WEEK);
  const [weekDays, setWeekDays] = useState(START_DAYS);
  const [hourIdx, setHourIdx] = useState(0);
  const [week, setWeek] = useState<WeekWeather | null>(null);
  const [weekVersion, setWeekVersion] = useState(0);
  const weekSpec = useMemo<WeekSpec>(() => ({ start: weekStart, days: weekDays }), [weekStart, weekDays]);
  useEffect(() => {
    if (!source || timeMode !== "week") return;
    const w = new WeekWeather(source.arrays, weekSpec, windMode, source.initTimeCount);
    const l = () => setWeekVersion(w.version);
    w.listeners.add(l);
    setWeek(w);
    setHourIdx((h) => Math.min(h, w.depth - 1));
    nextHourRef.current = (h) => w.nextHour(h);
    needWeatherRef.current = (px, py) => w.ensure(px, py);
    // tiles already on hand (a switch from the run, or another week): their shards for this week
    const live = aefLayersRef.current?.live;
    if (live) w.setView(Array.from(live.values(), (t) => t.grid));
    return () => {
      needWeatherRef.current = () => {};
      nextHourRef.current = (h) => (h + 1) % depthRef.current;
      w.listeners.delete(l);
      w.abort();
      setWeek(null);
    };
  }, [source, timeMode, weekSpec, windMode]);
  // the playhead moved: its day goes first in the week's queue
  useEffect(() => {
    week?.focus(hourIdx);
  }, [week, hourIdx]);
  // the view moved: the shards under the tiles on screen are the view (first in the queue, the loading bar)
  useEffect(() => {
    const live = aefLayersRef.current?.live;
    if (week && live) week.setView(Array.from(live.values(), (t) => t.grid));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [week, view, aefVersion]);

  // The ground needs the forecast under every tile it holds, whether or not the field is drawn:
  // a side channel fetches the shards the tiles ask for (it skips cached shards).
  useEffect(() => {
    if (!source || timeMode !== "run") return;
    const sc = new SideChannel(source.arrays, cacheRef.current);
    needWeatherRef.current = (px, py) => void sc.ensure(px, py, initTimeIdx, windMode);
    // tiles already on hand asked under an older run
    if (aefLayers) for (const t of aefLayers.live.values()) void sc.ensure(t.grid.px, t.grid.py, initTimeIdx, windMode);
    return () => {
      needWeatherRef.current = () => {};
      sc.abort();
    };
  }, [source, timeMode, initTimeIdx, windMode, aefLayers]);
  // The weather changed: the weather texture anew for the tiles it touches (a run shard: every tile).
  useEffect(() => {
    aefLayers?.refreshWeather();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aefLayers, cacheVersion]);
  useEffect(() => {
    if (!week || !aefLayers) return;
    const keys = week.takeArrived();
    if (keys.size > 0) aefLayers.refreshWeather(keys);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aefLayers, week, weekVersion]);
  /** The layer of the weather texture on screen: the lead hour of the run or the hour of the week. */
  const timeIdx = timeMode === "week" ? hourIdx : leadTimeIdx;
  const timeDepth = timeMode === "week" ? weekDays * HOURS_PER_DAY : HRRR_LEAD_TIME_COUNT;
  /** Where the tiles' weather textures read from, and a key that changes when that does. */
  const runSource = useMemo(() => cacheSource(cacheRef.current, fuel === "ignition" ? "ignition" : "ffwi"), [fuel]);
  const weatherSrc = timeMode === "week" && week ? week : runSource;
  timeModeRef.current = timeMode;
  depthRef.current = timeDepth;
  const weatherKey = timeMode === "week" && week ? `week|${week.id}` : `${fuel === "ignition" ? "ignition" : "ffwi"}|run`;

  // ---- hover: lon/lat -> class under the cursor and FFWI at the HRRR pixel
  const [hover, setHover] = useState<{ lon: number; lat: number; x: number; y: number } | null>(null);
  const hoverInfo = useMemo(() => {
    if (!hover) return null;
    const px = lonLatToHrrrPixel(hover.lon, hover.lat);
    const inWeek = timeMode === "week";
    const wx = px && !inWeek ? cacheRef.current.sample(px.x, px.y, leadTimeIdx) : null;
    const ground = aefLayers?.at(hover.lon, hover.lat) ?? null;
    const ci = ground ? (teacher === "nlcd" ? ground.nlcd : ground.esri) : NO_CLASS;
    const className = ci === NO_CLASS ? null : HEADS[teacher].names[ci]!;
    const weight = ci === NO_CLASS ? 0 : fuelOf(teacher, ci);
    const ffwi = inWeek ? (px && week ? week.ffwiAt(px.x, px.y, hourIdx) : NaN) : (wx?.ffwi ?? NaN);
    const wl = px
      ? inWeek
        ? (week ? week.termAt(px.x, px.y, hourIdx) : NaN)
        : cacheRef.current.ignitionAt(px.x, px.y, leadTimeIdx)
      : NaN;
    const wx0 = wl;
    const ignP = ground && Number.isFinite(wl) ? 1 / (1 + Math.exp(-(ground.ign + wl))) : NaN;
    const groundP = ground ? 1 / (1 + Math.exp(-ground.ign)) : NaN;
    return { wx, ffwi, ground, className, weight, lit: Number.isFinite(ffwi) ? ffwi * weight : NaN, ignP, groundP, wx0 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hover, leadTimeIdx, hourIdx, timeMode, week, cacheVersion, weekVersion, aefLayers, aefVersion, teacher]);

  // ---- the rank scale: the view's quantiles of the logit this hour set the logit ramp
  const rankRamp = useMemo<[number, number] | null>(() => {
    if (ignScale !== "rank" || !view || !aefLayers) return null;
    const kind = zoom < MOSAIC_MIN_ZOOM ? "coarse" : "zarr";
    const q = aefLayers.quantiles(view, timeIdx, [qLo, qHi], useWeather, kind);
    return q ? [q[0]!, Math.max(q[1]!, q[0]! + 0.05)] : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ignScale, view, zoom, timeIdx, qLo, qHi, useWeather, aefLayers, aefVersion]);

  // ---- contours of the field: recomputed when the hour, the weather, the view or the lag changes
  const contourLines = useMemo<ContourPath[]>(() => {
    if (!view || contourStep <= 0) return [];
    const levels: number[] = [];
    for (let v = Math.max(contourStep, hideBelow); v <= 200; v += contourStep) levels.push(v);
    return ffwiContours(cacheRef.current, leadTimeIdx, view, levels);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, contourStep, hideBelow, leadTimeIdx, cacheVersion, tauHours]);

  // ---- layers
  const selection = useMemo(() => buildSelection(initTimeIdx), [initTimeIdx]);
  const getTileData = useMemo(
    () => (source ? makeGetTileData({ arrays: source.arrays, cache: cacheRef.current, windMode, variable: rasterVariable, initTimeIdx }) : null),
    [source, windMode, rasterVariable, initTimeIdx],
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

  const deckLayers = useMemo(() => {
    const out: unknown[] = [];
    if (showRaster && source && getTileData && colormapTexture) {
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
    if (aefLayers) {
      out.push(
        ...aefLayers.layers({
          mode: fuel === "cover" ? "cover" : fuel === "ignition" ? "ignition" : "lit",
          zoom,
          teacher,
          leadTimeIdx: timeIdx,
          weather: weatherSrc,
          weatherKey,
          hideBelow,
          rescaleMax,
          pHide,
          pMax,
          scale: ignScale === "probability" ? "probability" : "logit",
          zLo: ignScale === "rank" ? (rankRamp ? rankRamp[0] : 99) : zLo,
          zHi: ignScale === "rank" ? (rankRamp ? rankRamp[1] : 100) : zHi,
          useWeather,
          standingAlpha,
          wxGate,
          glowHours,
          grain,
          light,
          opacity: fuelOpacity,
          visible: fuel !== "off",
          beforeId: BEFORE_ID,
        }),
      );
    }
    // if (runPerimeters.length > 0) out.push(perimeterLayer({ perimeters: runPerimeters, initTime: dateFromInitTimeIdx(initTimeIdx), leadHours: leadTimeIdx, visible: showPerimeters, beforeId: BEFORE_ID }));
    // if (fireList.length > 0) out.push(...fireLayers({ fires: fireList, initTime: dateFromInitTimeIdx(initTimeIdx), leadHours: leadTimeIdx, visible: showFires, beforeId: BEFORE_ID }));
    if (contourLines.length > 0) {
      out.push(
        new PathLayer<ContourPath>({
          id: "ffwi-contours",
          data: contourLines,
          getPath: (d: ContourPath) => d.path,
          getColor: (d: ContourPath) => [226, 236, 255, Math.min(255, 90 + d.level * 1.4)],
          getWidth: (d: ContourPath) => 0.6 + d.level / 60,
          widthUnits: "pixels",
          jointRounded: true,
          capRounded: true,
          beforeId: BEFORE_ID,
          updateTriggers: { getColor: [contourLines], getWidth: [contourLines] },
        } as never),
      );
    }
    return out;
    // aefVersion: the controller's tiles or weather changed under the same props
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    showRaster, source, getTileData, colormapTexture, initTimeIdx, windMode, rasterVariable, selection, renderTile, rasterOpacity,
    leadTimeIdx, colormapIndex, hideBelow, rescaleMax, tauHours, aefLayers, aefVersion, fuel, teacher, fuelOpacity, zoom,
    glowHours, grain, light, contourLines, pHide, pMax, ignScale, zLo, zHi, rankRamp, useWeather, standingAlpha, wxGate,
    timeIdx, weatherSrc, weatherKey, timeMode,
  ]);

  const status: Status = {
    backend: source?.backend ?? null,
    tilesLoaded: cacheRef.current.size,
    tilesTotal: Math.max(cacheRef.current.size, showRaster ? 24 : 0),
    aefIndexReady: aefIndex !== null && mosaic !== null,
    aefReading: aefLayers?.work.reading ?? 0,
    aefDone: aefLayers?.work.done ?? 0,
    weekDone: timeMode === "week" && week ? week.progress.done : 0,
    weekWanted: timeMode === "week" && week ? week.progress.wanted : 0,
    weekDaysIn: timeMode === "week" && week ? week.daysReady() : 0,
    weekDaysTotal: timeMode === "week" ? weekDays : 0,
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

  const runLabel = source ? dateFromInitTimeIdx(initTimeIdx).toISOString().slice(0, 13).replace("T", " ") + "Z" : "";
  const weekHourLabel = weekHourDate(weekSpec, hourIdx).toISOString().slice(0, 13).replace("T", " ") + "Z";

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <MaplibreMap
        ref={mapRef}
        initialViewState={START_VIEW}
        hash
        mapStyle={MAP_STYLE_URL}
        attributionControl={false}
        onLoad={(e) => {
          setZoom(e.target.getZoom());
          addHillshade(e.target, relief);
          readView(e.target);
        }}
        onMove={(e) => setZoom(e.viewState.zoom)}
        onMoveEnd={(e) => readView(e.target)}
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
        timeMode={timeMode}
        onTimeModeChange={setTimeMode}
        weekStart={weekStart}
        onWeekStartChange={setWeekStart}
        weekDays={weekDays}
        onWeekDaysChange={setWeekDays}
        hourIdx={hourIdx}
        onHourIdxChange={setHourIdx}
        weekDepth={timeMode === "week" ? timeDepth : weekDays * HOURS_PER_DAY}
        weekHourLabel={weekHourLabel}
        weekDayStates={timeMode === "week" && week ? week.dayStates() : []}
        fuel={fuel}
        onFuelChange={setFuel}
        pHide={pHide}
        onPHideChange={setPHide}
        pMax={pMax}
        onPMaxChange={setPMax}
        ignScale={ignScale}
        onIgnScaleChange={setIgnScale}
        zLo={zLo}
        onZLoChange={setZLo}
        zHi={zHi}
        onZHiChange={setZHi}
        qLo={qLo}
        onQLoChange={setQLo}
        qHi={qHi}
        onQHiChange={setQHi}
        rankRamp={rankRamp}
        useWeather={useWeather}
        onUseWeatherChange={setUseWeather}
        standingAlpha={standingAlpha}
        onStandingAlphaChange={setStandingAlpha}
        wxGate={wxGate}
        onWxGateChange={setWxGate}
        wxNow={hoverInfo && Number.isFinite(hoverInfo.wx0) ? hoverInfo.wx0 : null}


        teacher={teacher}
        onTeacherChange={setTeacher}
        fuelOpacity={fuelOpacity}
        onFuelOpacityChange={setFuelOpacity}
        glowHours={glowHours}
        onGlowHoursChange={setGlowHours}
        grain={grain}
        onGrainChange={setGrain}
        light={light}
        onLightChange={setLight}
        relief={relief}
        onReliefChange={setRelief}
        contourStep={contourStep}
        onContourStepChange={setContourStep}
        zoom={zoom}
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
        rasterVariable={rasterVariable}
        onRasterVariableChange={setRasterVariable}
        rasterOpacity={rasterOpacity}
        onRasterOpacityChange={setRasterOpacity}
        tauHours={tauHours}
        onTauHoursChange={setTauHours}
        status={status}
      />
      {hover && hoverInfo && (hoverInfo.className || Number.isFinite(hoverInfo.ffwi)) ? (
        <div style={tooltipStyle(hover.x, hover.y)}>
          <div style={{ fontWeight: 600 }}>
            {hoverInfo.ground
              ? `ignition ${Number.isFinite(hoverInfo.ignP) ? `${Math.round(hoverInfo.ignP * 100)}%` : "?"} this hour · ground alone ${Math.round(hoverInfo.groundP * 100)}%${Number.isFinite(hoverInfo.wx0) ? ` · weather term ${hoverInfo.wx0 >= 0 ? "+" : ""}${hoverInfo.wx0.toFixed(1)}` : ""}`
              : "no embedding here"}
          </div>
          <div style={{ color: "#bbb" }}>
            {hoverInfo.className ? `reads as ${hoverInfo.className}` : ""}
            {hoverInfo.ground ? ` · canopy ${Math.round(hoverInfo.ground.cover * 100)}%` : ""}
          </div>
          <div>
            {hoverInfo.wx && Number.isFinite(hoverInfo.wx.ffwi)
              ? `FFWI ${hoverInfo.wx.ffwi.toFixed(0)} · wind ${hoverInfo.wx.windMph} mph · RH ${hoverInfo.wx.rh}% · ${hoverInfo.wx.tc} °C`
              : Number.isFinite(hoverInfo.ffwi)
                ? `FFWI ${hoverInfo.ffwi.toFixed(0)}`
                : "forecast not loaded here yet"}
            {tauHours > 0 ? ` (fuel lag ${tauHours} h)` : ""}
          </div>
          <div style={{ color: "#aaa" }}>
            {hover.lat.toFixed(3)}, {hover.lon.toFixed(3)} ·{" "}
            {timeMode === "week" ? `${weekHourLabel} (12Z run, +${hourIdx % HOURS_PER_DAY} h)` : `${runLabel} +${leadTimeIdx} h`}
          </div>
        </div>
      ) : null}
    </div>
  );
}
