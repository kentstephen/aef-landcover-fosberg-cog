import { COLORMAP_INDEX } from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { useState } from "react";
import type { WindMode } from "../ffwi/compute.js";
import {
  dateFromInitTimeIdx,
  HRRR_LEAD_TIME_COUNT,
  HRRR_LEAD_TIME_HOURS,
  INIT_TIME_ORIGIN,
  initTimeIdxFromDate,
  isoDateString,
} from "../hrrr/metadata.js";
import { RASTER_VARIABLES, windLabel, type RasterVariable } from "../hrrr/variables.js";
import { POTENTIAL_LAYERS, type PotentialLayerId } from "../potential/service.js";
import { A, Check, Field, Fold, Help, Ramp, Seg, Slider } from "./controls.js";
import "./panel.css";

const COLORMAP_ROW_COUNT = Object.keys(COLORMAP_INDEX).length;

/** Protan-safe, luminance-monotone ramps with a bright hot end. */
export const COLORMAP_CHOICES = [
  { id: "inferno", label: "Inferno" },
  { id: "magma", label: "Magma" },
  { id: "thermal", label: "Thermal" },
  { id: "solar", label: "Solar" },
  { id: "hot", label: "Hot" },
  { id: "afmhot", label: "Afmhot" },
  { id: "viridis", label: "Viridis" },
  { id: "cividis", label: "Cividis" },
  { id: "ylorbr", label: "YlOrBr" },
  { id: "coolwarm", label: "Cool-warm (diverging)" },
] as const;
export type ColormapChoiceId = (typeof COLORMAP_CHOICES)[number]["id"];

/** What the map draws: one FFWI number per county (default), or the 3 km raster of FFWI or one of its inputs. */
export type ViewMode = "county" | "raster";
export type CountyField = "ffwi" | "peak" | "bp";
export type PeakMode = "peak" | "onset";
export type CountyStat = "share" | "mean" | "max" | "hours";

export type Status = {
  backend: "icechunk" | "zarr-v3" | null;
  tilesLoaded: number;
  tilesTotal: number;
  potentialReady: boolean;
  potentialLoading: boolean;
  /** County view only: the counties GeoJSON + join. */
  countiesNeeded: boolean;
  countiesReady: boolean;
  linesNeeded: boolean;
  linesReady: boolean;
  error: string | null;
};

/** The headline number: miles of transmission line above the hide line this hour, and the run's peak. */
export type LineHeadline = {
  milesNow: number;
  milesTotal: number;
  milesLoaded: number;
  peakLead: number;
  milesPeak: number;
};

/** What is still loading, for the panel bar and the map pill. */
export function loadingSummary(s: Status): { fraction: number; label: string } | null {
  const parts: string[] = [];
  let done = 0;
  let total = 0;
  if (!s.backend) {
    parts.push("opening the HRRR store");
    total += 1;
  }
  total += s.tilesTotal;
  done += Math.min(s.tilesLoaded, s.tilesTotal);
  if (s.tilesLoaded < s.tilesTotal) parts.push(`forecast tiles ${s.tilesLoaded} / ${s.tilesTotal}`);
  if (s.countiesNeeded) {
    total += 1;
    if (s.countiesReady) done += 1;
    else parts.push("counties");
  }
  if (s.linesNeeded) {
    total += 1;
    if (s.linesReady) done += 1;
    else parts.push("transmission lines");
  }
  total += 1;
  if (s.potentialReady) done += 1;
  else parts.push("burn probability");
  if (parts.length === 0) return null;
  return { fraction: total > 0 ? done / total : 0, label: `Loading ${parts.join(" · ")}` };
}

export type ControlPanelProps = {
  leadTimeIdx: number;
  initTimeIdx: number;
  initTimeCount: number;
  isPlaying: boolean;
  frameDurationMs: number;
  onInitTimeIdxChange: (idx: number) => void;
  onLeadTimeIdxChange: (idx: number) => void;
  onPlayPauseToggle: () => void;
  onFrameDurationMsChange: (ms: number) => void;

  windMode: WindMode;
  onWindModeChange: (m: WindMode) => void;
  colormapId: ColormapChoiceId;
  onColormapChange: (id: ColormapChoiceId) => void;
  rescaleMax: number;
  onRescaleMaxChange: (v: number) => void;
  hideBelow: number;
  onHideBelowChange: (v: number) => void;
  showRaster: boolean;
  onShowRasterChange: (v: boolean) => void;
  showCounties: boolean;
  onShowCountiesChange: (v: boolean) => void;
  rasterOpacity: number;
  onRasterOpacityChange: (v: number) => void;

  potentialId: PotentialLayerId;
  onPotentialIdChange: (id: PotentialLayerId) => void;
  potentialNorm: number;
  onPotentialNormChange: (v: number) => void;
  viewMode: ViewMode;
  onViewModeChange: (m: ViewMode) => void;
  rasterVariable: RasterVariable;
  onRasterVariableChange: (v: RasterVariable) => void;
  countyField: CountyField;
  onCountyFieldChange: (f: CountyField) => void;
  countyStat: CountyStat;
  onCountyStatChange: (s: CountyStat) => void;
  peakMode: PeakMode;
  onPeakModeChange: (m: PeakMode) => void;
  countyCount: number;
  tauHours: number;
  onTauHoursChange: (v: number) => void;
  windowHours: number;
  onWindowHoursChange: (v: number) => void;

  showCanopy: boolean;
  onShowCanopyChange: (v: boolean) => void;
  canopyReady: boolean;
  canopyOpacity: number;
  onCanopyOpacityChange: (v: number) => void;
  canopyMinHeight: number;
  onCanopyMinHeightChange: (v: number) => void;
  canopyMaxHeight: number;
  onCanopyMaxHeightChange: (v: number) => void;
  showLines: boolean;
  onShowLinesChange: (v: boolean) => void;
  lineGhost: number;
  onLineGhostChange: (v: number) => void;
  lineMaxWidth: number;
  onLineMaxWidthChange: (v: number) => void;
  lineHeadline: LineHeadline | null;

  status: Status;
};

function fmtMiles(mi: number): string {
  return mi >= 10_000 ? `${(mi / 1000).toFixed(0)}k` : mi >= 1000 ? `${(mi / 1000).toFixed(1)}k` : `${Math.round(mi)}`;
}

function fmtRun(init: Date, leadH: number): string {
  const d = new Date(init.getTime() + leadH * 3_600_000);
  return `${String(d.getUTCDate()).padStart(2, "0")} ${String(d.getUTCHours()).padStart(2, "0")}Z`;
}

const UNDER_OPTIONS = [
  { id: "nothing", label: "nothing" },
  ...Object.values(RASTER_VARIABLES).map((v) => ({ id: v.id as string, label: v.id === "ffwi" ? "FFWI raster" : v.label.toLowerCase() })),
  { id: "county", label: "counties" },
];

export function ControlPanel(p: ControlPanelProps) {
  const hours = HRRR_LEAD_TIME_HOURS[p.leadTimeIdx] ?? 0;
  const validUTC = dateFromInitTimeIdx(p.initTimeIdx);
  validUTC.setUTCHours(validUTC.getUTCHours() + hours);
  const validStr = `${isoDateString(validUTC)} ${String(validUTC.getUTCHours()).padStart(2, "0")}:00Z`;
  const s = p.status;
  const loading = loadingSummary(s);
  const pot = POTENTIAL_LAYERS[p.potentialId];
  const rv = RASTER_VARIABLES[p.rasterVariable];
  const [min, setMin] = useState(false);

  const under = p.viewMode === "county" ? (p.showCounties ? "county" : "nothing") : p.showRaster ? p.rasterVariable : "nothing";
  const underOptions = UNDER_OPTIONS.map((o) => (o.id === "wind" ? { ...o, label: windLabel(p.windMode).replace(/^wind: /, "wind ") } : o));
  const setUnder = (id: string) => {
    if (id === "nothing") {
      p.onShowRasterChange(false);
      p.onShowCountiesChange(false);
    } else if (id === "county") {
      p.onViewModeChange("county");
      p.onShowCountiesChange(true);
    } else {
      p.onViewModeChange("raster");
      p.onShowRasterChange(true);
      p.onRasterVariableChange(id as RasterVariable);
    }
  };

  const rampLabel =
    p.viewMode === "county"
      ? p.countyField === "peak"
        ? `${p.peakMode === "peak" ? "hour of peak" : "first hour ≥ " + p.hideBelow}: +0 h (${fmtRun(dateFromInitTimeIdx(p.initTimeIdx), 0)}) to +48 h (${fmtRun(dateFromInitTimeIdx(p.initTimeIdx), 48)})`
        : p.countyField === "bp"
          ? `county mean ${pot.shortLabel} 0 to ${pot.format(p.potentialNorm)} (sqrt)`
          : p.countyStat === "share"
            ? `share of pixels ≥ ${p.hideBelow}: 0 to 100%`
            : p.countyStat === "hours"
              ? `hours ≥ ${p.hideBelow} in the last ${p.windowHours}: 0 to ${p.windowHours}`
              : `county ${p.countyStat} FFWI ${p.hideBelow} to ${p.rescaleMax}`
      : p.rasterVariable === "ffwi"
        ? `FFWI${p.tauHours > 0 ? ` (${p.tauHours} h fuel lag)` : ""} ${p.hideBelow} to ${p.rescaleMax}`
        : `${rv.format(rv.min)} to ${rv.format(rv.max)}${rv.reversed ? " (reversed: dry is bright)" : ""}`;

  return (
    <div className={`pn${min ? " min" : ""}`}>
      <div className="pn-title">
        <h1>Fire weather, hour by hour</h1>
        <button type="button" className="pn-min" aria-label={min ? "Expand panel" : "Collapse panel"} onClick={() => setMin((v) => !v)}>
          {min ? "▸" : "▾"}
        </button>
      </div>

      <p>
        The <b>Fosberg Fire Weather Index</b> is one number for how hard the weather would push a fire in dry grass or brush:
        temperature and humidity set how dry fine fuel is, wind does the escalating; bone-dry fuel in a 30 mph wind = 100. It knows
        nothing about the ground. Computed here from NOAA's HRRR forecast at 3 km, one frame per hour for the next two days. The{" "}
        <b>county view</b> shows one FFWI number per county this hour; hover for the county's USFS burn probability, or press the
        button for a static map of it. The <b>raster view</b> shows the 3 km field itself: FFWI or any of its inputs (temperature,
        humidity, wind).
      </p>
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: <A href="https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/">NOAA HRRR 48 h</A> via{" "}
        <A href="https://dynamical.org">dynamical.org</A> ({s.backend === "icechunk" ? "Icechunk" : s.backend === "zarr-v3" ? "Zarr v3" : "opening"}) ·{" "}
        <A href="https://wildfirerisk.org">USFS Wildfire Risk to Communities 2024</A> via{" "}
        <A href="https://imagery.geoplatform.gov/iipp/rest/services/Fire_Aviation">GeoPlatform</A> ·{" "}
        <A href="https://www.frames.gov/catalog/27085">Fosberg 1978</A> ·{" "}
        <A href="https://hifld-geoplatform.hub.arcgis.com/datasets/electric-power-transmission-lines">HIFLD transmission lines</A> ·{" "}
        <A href="https://source.coop/tge-labs/meta-chm-v2">Meta / WRI canopy height v2</A>
      </p>

      {loading ? (
        <div className="field" style={{ marginTop: 8 }}>
          <div className="bar"><span style={{ width: `${Math.round(loading.fraction * 100)}%` }} /></div>
          <p className="muted small">{loading.label}. The forecast is read straight from the cloud store, about 1 MB per tile; the county map fills in as tiles arrive.</p>
        </div>
      ) : null}

      <Fold k="time" label="Forecast hour" defaultOpen tag={`+${hours} h · ${validStr}`}>
        <Field label="Forecast run (UTC)">
          <div className="row">
            <input
              type="date"
              min={isoDateString(INIT_TIME_ORIGIN)}
              max={p.initTimeCount > 0 ? isoDateString(dateFromInitTimeIdx(p.initTimeCount - 1)) : undefined}
              value={isoDateString(dateFromInitTimeIdx(p.initTimeIdx))}
              disabled={p.initTimeCount === 0}
              onChange={(e) => p.onInitTimeIdxChange(initTimeIdxFromDate(new Date(`${e.target.value}T00:00:00Z`), Math.max(0, p.initTimeCount - 1)))}
            />
            {p.initTimeCount > 0 && p.initTimeIdx < p.initTimeCount - 1 ? (
              <a href="#" className="small" onClick={(e) => { e.preventDefault(); p.onInitTimeIdxChange(p.initTimeCount - 1); }}>Jump to latest forecast</a>
            ) : (
              <span className="muted small">Latest forecast run</span>
            )}
          </div>
        </Field>
        <Field
          label={
            <span className="row">
              <span>+<span style={{ display: "inline-block", minWidth: "2ch", textAlign: "right" }}>{hours}</span> h · {validStr}</span>
              <Help label="Forecast hour">49 hourly steps, +0 to +48 h. Space toggles play.</Help>
            </span>
          }
        >
          <div className="row" style={{ flexWrap: "nowrap" }}>
            <button type="button" className="play" aria-label={p.isPlaying ? "Pause" : "Play"} onClick={(e) => { p.onPlayPauseToggle(); e.currentTarget.blur(); }}>
              {p.isPlaying ? "❚❚" : "▶"}
            </button>
            <Slider min={0} max={HRRR_LEAD_TIME_COUNT - 1} step={1} value={p.leadTimeIdx} onChange={p.onLeadTimeIdxChange} label="Forecast hour" onReset={() => p.onLeadTimeIdxChange(0)} />
          </div>
        </Field>
        <Field label={`Seconds per forecast hour: ${(p.frameDurationMs / 1000).toFixed(2)}`}>
          <Slider min={80} max={1000} step={20} value={p.frameDurationMs} onChange={p.onFrameDurationMsChange} label="Frame duration" onReset={() => p.onFrameDurationMsChange(300)} />
        </Field>
      </Fold>

      <Fold k="lines" label="Transmission lines" defaultOpen tag={p.showLines ? "on" : "off"}>
        <div className="row">
          <Check checked={p.showLines} onChange={p.onShowLinesChange} label="Show" />
          <Help label="Lines">
            HIFLD Electric Power Transmission Lines (52k lines at 100 kV and up, about 420k miles), cut into pieces of at most 3 km
            and lit by the FFWI at the HRRR pixel under each piece. Public safety power shutoffs are called on these same inputs
            (wind and low humidity), and line faults are a leading ignition cause. Faint = below the hide line; above it the line
            takes the colormap from the hide line to the ramp top and widens with it. Voltage class is in the tooltip.
          </Help>
        </div>
        {p.showLines ? (
          <>
            <p className="lead">
              {p.lineHeadline
                ? `${fmtMiles(p.lineHeadline.milesNow)} of ${fmtMiles(p.lineHeadline.milesTotal)} miles of transmission line above FFWI ${p.hideBelow} at +${hours} h` +
                  (p.lineHeadline.milesPeak > 0 ? `, peaking at ${fmtMiles(p.lineHeadline.milesPeak)} miles at +${p.lineHeadline.peakLead} h` : "") +
                  (p.lineHeadline.milesLoaded < 0.98 * p.lineHeadline.milesTotal
                    ? ` (${Math.round((100 * p.lineHeadline.milesLoaded) / p.lineHeadline.milesTotal)}% of the network has its forecast so far)`
                    : "")
                : s.linesReady
                  ? "waiting for the forecast tiles"
                  : "loading the lines (9 MB)"}
            </p>
            <Field label={`Width: 1 px at FFWI ${p.hideBelow} to ${p.lineMaxWidth} px at ${p.rescaleMax} (color and width follow the same number; the colormap is the one below, used from its 35% mark up so a lit line is never the dark end)`}>
              <Slider min={1} max={6} step={0.5} value={p.lineMaxWidth} onChange={p.onLineMaxWidthChange} label="Max width" onReset={() => p.onLineMaxWidthChange(3)} />
            </Field>
            <Field label={`Lines below the hide line at ${Math.round(p.lineGhost * 100)}% opacity${p.lineGhost === 0 ? " (hidden)" : ""}`}>
              <Slider min={0} max={0.6} step={0.01} value={p.lineGhost} onChange={p.onLineGhostChange} label="Ghost opacity" onReset={() => p.onLineGhostChange(0.18)} />
            </Field>
          </>
        ) : null}
      </Fold>

      <Fold k="canopy" label="Canopy" tag={p.showCanopy ? "on" : "off"}>
        <div className="row">
          <Check checked={p.showCanopy} onChange={p.onShowCanopyChange} label="Show" />
          <Help label="Canopy">
            Meta and WRI global tree canopy height v2 (DINOv3 on satellite imagery, about 1.2 m native), drawn here from its global
            overview as a single green ramp: dark at the minimum height, bright at the maximum, nothing drawn below the minimum.
            Context for the lines, not a second subject: a treed corridor is where a tree can reach a conductor in wind.
          </Help>
        </div>
        {p.showCanopy ? (
          <>
            {!p.canopyReady ? <p className="muted small">opening the canopy overview</p> : null}
            <Field label={`Opacity ${Math.round(p.canopyOpacity * 100)}%`}>
              <Slider min={0.05} max={1} step={0.05} value={p.canopyOpacity} onChange={p.onCanopyOpacityChange} label="Canopy opacity" onReset={() => p.onCanopyOpacityChange(0.3)} />
            </Field>
            <Field label={`Drawn from ${p.canopyMinHeight} m, brightest at ${p.canopyMaxHeight} m`}>
              <Slider min={1} max={20} step={1} value={p.canopyMinHeight} onChange={p.onCanopyMinHeightChange} label="Min height" onReset={() => p.onCanopyMinHeightChange(3)} />
              <Slider min={10} max={60} step={5} value={p.canopyMaxHeight} onChange={p.onCanopyMaxHeightChange} label="Max height" onReset={() => p.onCanopyMaxHeightChange(30)} />
            </Field>
          </>
        ) : null}
      </Fold>

      <Fold k="field" label="The field" defaultOpen tag={underOptions.find((o) => o.id === under)?.label}>
        <div className="row">
          <span className="muted small">Under the lines:</span>
          <Help label="FFWI">
            Fosberg 1978. Temperature and humidity set how dry fine fuel is; wind does the escalating. Sustained 10 m wind is the
            standard input. 3 km HRRR pixels, reprojected on the GPU.
          </Help>
        </div>
        <Seg value={under} options={underOptions} onChange={setUnder} />
        {p.viewMode === "county" && p.showCounties ? (
          <>
            <div className="row">
              <Seg
                value={p.countyField}
                options={[
                  { id: "ffwi", label: "FFWI this hour" },
                  { id: "peak", label: "When it peaks" },
                  { id: "bp", label: "Burn probability" },
                ]}
                onChange={p.onCountyFieldChange}
              />
              <span className="muted small">{p.countyCount ? `${p.countyCount} counties` : "loading counties"}</span>
            </div>
            {p.countyField === "ffwi" ? (
              <select value={p.countyStat} onChange={(e) => p.onCountyStatChange(e.target.value as CountyStat)}>
                <option value="share">share of the county's pixels with FFWI ≥ hide line</option>
                <option value="mean">mean FFWI over the county</option>
                <option value="max">max FFWI in the county</option>
                <option value="hours">hours in the trailing window with FFWI ≥ hide line</option>
              </select>
            ) : null}
            {p.countyField === "peak" ? (
              <select value={p.peakMode} onChange={(e) => p.onPeakModeChange(e.target.value as PeakMode)}>
                <option value="peak">hour when the county's mean FFWI peaks (whole 48 h, static)</option>
                <option value="onset">first hour the county's mean FFWI reaches the hide line</option>
              </select>
            ) : null}
            {p.countyField === "ffwi" && p.countyStat === "hours" ? (
              <Field label={`Trailing window: ${p.windowHours} h (counts back to the run's +0 h at most)`}>
                <Slider min={3} max={48} step={1} value={p.windowHours} onChange={p.onWindowHoursChange} label="Trailing window" onReset={() => p.onWindowHoursChange(12)} />
              </Field>
            ) : null}
          </>
        ) : null}
        <div className="row" style={{ flexWrap: "nowrap" }}>
          <select value={p.colormapId} onChange={(e) => p.onColormapChange(e.target.value as ColormapChoiceId)} style={{ flex: "0 0 auto" }}>
            {COLORMAP_CHOICES.map((c) => (<option key={c.id} value={c.id}>{c.label}</option>))}
          </select>
          <Ramp spriteUrl={colormapsPngUrl} rowCount={COLORMAP_ROW_COUNT} rowIndex={COLORMAP_INDEX[p.colormapId]!} label={rampLabel} />
        </div>
        <p className="muted small">{rampLabel}</p>
        <Field label={`Opacity ${Math.round(p.rasterOpacity * 100)}%`}>
          <Slider min={0} max={1} step={0.01} value={p.rasterOpacity} onChange={p.onRasterOpacityChange} label="Layer opacity" onReset={() => p.onRasterOpacityChange(0.7)} />
        </Field>
        <Field label={p.hideBelow > 0 ? `Hide FFWI below ${p.hideBelow} (the map shows through; the county "share" threshold)` : "Showing every pixel"}>
          <Slider min={0} max={80} step={1} value={p.hideBelow} onChange={p.onHideBelowChange} label="Hide below" onReset={() => p.onHideBelowChange(20)} />
        </Field>
        <Field label={`FFWI that counts as fully pushing (ramp top): ${p.rescaleMax}`}>
          <Slider min={40} max={200} step={5} value={p.rescaleMax} onChange={p.onRescaleMaxChange} label="Rescale max" onReset={() => p.onRescaleMaxChange(100)} />
        </Field>
        <Field
          label={
            <span className="row">
              <span>
                {p.tauHours === 0
                  ? "Fuel moisture lag: 0 h (Fosberg as published, fuel dries and wets instantly)"
                  : `Fuel moisture lag: ${p.tauHours} h (memory reaches back ${Math.min(p.leadTimeIdx, 3 * p.tauHours)} of ~${3 * p.tauHours} h at this frame)`}
              </span>
              <Help label="Timelag">
                Fosberg uses the equilibrium moisture the air implies right now. Real dead fuel lags behind it: the standard NFDRS
                form is m(t) = m_eq(t) + (m(t-1) - m_eq(t)) x exp(-1/tau), with tau about 1 h for fine fuels and 10 h for the
                10-hour class. With tau set, the moisture term carries that memory (seeded at the run's +0 h) and wind stays
                instantaneous. The county stats, the raster and the tooltip all use the lagged value.
              </Help>
            </span>
          }
        >
          <Slider min={0} max={24} step={1} value={p.tauHours} onChange={p.onTauHoursChange} label="Fuel moisture lag" onReset={() => p.onTauHoursChange(0)} />
        </Field>
        <select value={p.windMode} onChange={(e) => p.onWindModeChange(e.target.value as WindMode)}>
          <option value="sustained">wind: 10 m sustained (standard FFWI)</option>
          <option value="gust">wind: gusts (higher; not comparable to FFWI tables)</option>
        </select>
      </Fold>

      <Fold k="land" label="The land (burn probability)">
        <div className="row">
          <select value={p.potentialId} onChange={(e) => p.onPotentialIdChange(e.target.value as PotentialLayerId)}>
            {Object.values(POTENTIAL_LAYERS).map((l) => (<option key={l.id} value={l.id}>{l.label}</option>))}
          </select>
          <Help label="Potential layer">{pot.about} One CONUS-wide export, averaged per county and read under the cursor.</Help>
        </div>
        <Field label={`${pot.shortLabel} ramp top for the county map: ${pot.format(p.potentialNorm)}`}>
          <Slider min={pot.sliderStep} max={pot.displayMax} step={pot.sliderStep} value={p.potentialNorm} onChange={p.onPotentialNormChange} label="BP normalization" onReset={() => p.onPotentialNormChange(pot.id === "bp" ? 0.02 : 5000)} />
        </Field>
        <p className="muted small">
          {s.potentialLoading ? "fetching potential…" : s.potentialReady ? "potential live" : "potential not loaded"} · {s.tilesLoaded} / {s.tilesTotal} forecast tiles
        </p>
      </Fold>

      {s.error ? <p className="warn small" style={{ marginTop: 8 }}>{s.error}</p> : null}

      <div className="pn-foot">
        <span>Built with <A href="https://github.com/developmentseed/deck.gl-raster">deck.gl-raster</A> by <A href="https://developmentseed.org">Development Seed</A></span>
        <span>© <A href="https://carto.com/attributions">CARTO</A>, © <A href="https://www.openstreetmap.org/copyright">OpenStreetMap</A> contributors</span>
      </div>
    </div>
  );
}
