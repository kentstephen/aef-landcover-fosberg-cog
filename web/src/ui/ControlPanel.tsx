import { COLORMAP_INDEX } from "@developmentseed/deck.gl-raster/gpu-modules";
import colormapsPngUrl from "@developmentseed/deck.gl-raster/gpu-modules/colormaps.png";
import { useState } from "react";
import type { Teacher } from "../aef/head.js";
import { MOSAIC_MIN_ZOOM } from "../aef/mosaic.js";
import { rampGradient } from "../aef/ramp.js";
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
import { A, Check, Field, Fold, Help, Ramp, Seg, Slider } from "./controls.js";
import "./panel.css";

const COLORMAP_ROW_COUNT = Object.keys(COLORMAP_INDEX).length;

/** Protan-safe, luminance-monotone ramps with a bright hot end (the HRRR field). */
export const COLORMAP_CHOICES = [
  { id: "solar", label: "Solar" },
  { id: "inferno", label: "Inferno" },
  { id: "magma", label: "Magma" },
  { id: "thermal", label: "Thermal" },
  { id: "hot", label: "Hot" },
  { id: "viridis", label: "Viridis" },
  { id: "cividis", label: "Cividis" },
  { id: "ylorbr", label: "YlOrBr" },
  { id: "coolwarm", label: "Cool-warm (diverging)" },
] as const;
export type ColormapChoiceId = (typeof COLORMAP_CHOICES)[number]["id"];

/** What the land cover layer draws: the ground lit by the hour's FFWI, the classes themselves, or nothing. */
export type FuelChoice = "lit" | "cover" | "off";

export type Status = {
  backend: "icechunk" | "zarr-v3" | null;
  tilesLoaded: number;
  tilesTotal: number;
  aefIndexReady: boolean;
  aefReading: number;
  aefDone: number;
  error: string | null;
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
  if (s.tilesLoaded < s.tilesTotal) parts.push(`forecast ${s.tilesLoaded} / ${s.tilesTotal}`);
  if (!s.aefIndexReady) {
    total += 1;
    parts.push("the embedding index");
  }
  if (s.aefReading > 0) {
    total += s.aefReading + s.aefDone;
    done += s.aefDone;
    parts.push(`land cover ${s.aefDone} / ${s.aefReading + s.aefDone}`);
  }
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

  fuel: FuelChoice;
  onFuelChange: (f: FuelChoice) => void;
  teacher: Teacher;
  onTeacherChange: (t: Teacher) => void;
  fuelOpacity: number;
  onFuelOpacityChange: (v: number) => void;
  glowHours: number;
  onGlowHoursChange: (v: number) => void;
  grain: number;
  onGrainChange: (v: number) => void;
  light: boolean;
  onLightChange: (v: boolean) => void;
  relief: boolean;
  onReliefChange: (v: boolean) => void;
  contourStep: number;
  onContourStepChange: (v: number) => void;
  zoom: number;

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
  rasterVariable: RasterVariable;
  onRasterVariableChange: (v: RasterVariable) => void;
  rasterOpacity: number;
  onRasterOpacityChange: (v: number) => void;
  tauHours: number;
  onTauHoursChange: (v: number) => void;

  status: Status;
};

const FIELD_OPTIONS = [
  { id: "nothing", label: "nothing" },
  ...Object.values(RASTER_VARIABLES).map((v) => ({ id: v.id as string, label: v.id === "ffwi" ? "FFWI" : v.label.toLowerCase() })),
];

export function ControlPanel(p: ControlPanelProps) {
  const hours = HRRR_LEAD_TIME_HOURS[p.leadTimeIdx] ?? 0;
  const validUTC = dateFromInitTimeIdx(p.initTimeIdx);
  validUTC.setUTCHours(validUTC.getUTCHours() + hours);
  const validStr = `${isoDateString(validUTC)} ${String(validUTC.getUTCHours()).padStart(2, "0")}:00Z`;
  const s = p.status;
  const loading = loadingSummary(s);
  const rv = RASTER_VARIABLES[p.rasterVariable];
  const [min, setMin] = useState(false);
  const reader = p.zoom >= MOSAIC_MIN_ZOOM ? "10 m mosaic" : "COG overviews";

  const field = p.showRaster ? p.rasterVariable : "nothing";
  const fieldOptions = FIELD_OPTIONS.map((o) => (o.id === "wind" ? { ...o, label: windLabel(p.windMode).replace(/^wind: /, "wind ") } : o));
  const setField = (id: string) => {
    if (id === "nothing") p.onShowRasterChange(false);
    else {
      p.onShowRasterChange(true);
      p.onRasterVariableChange(id as RasterVariable);
    }
  };
  const fieldRampLabel =
    p.rasterVariable === "ffwi"
      ? `FFWI${p.tauHours > 0 ? ` (${p.tauHours} h fuel lag)` : ""} ${p.hideBelow} to ${p.rescaleMax}`
      : `${rv.format(rv.min)} to ${rv.format(rv.max)}${rv.reversed ? " (reversed: dry is bright)" : ""}`;

  return (
    <div className={`pn${min ? " min" : ""}`}>
      <div className="pn-title">
        <h1>Fuel under fire weather</h1>
        <button type="button" className="pn-min" aria-label={min ? "Expand panel" : "Collapse panel"} onClick={() => setMin((v) => !v)}>
          {min ? "▸" : "▾"}
        </button>
      </div>

      <p>
        The ground, read from <b>AlphaEarth embeddings</b> by a small land cover model, lit hour by hour by the{" "}
        <b>Fosberg Fire Weather Index</b> from NOAA's HRRR forecast. Open fuel (grass, shrub, pasture) carries the whole index,
        forest part of it, built-up ground and water none; where the weather is calm the ground is invisible, and it comes up
        in viridis as the forecast worsens.
      </p>
      <p className="muted small" style={{ marginTop: 6 }}>
        Data: <A href="https://source.coop/tge-labs/aef">AlphaEarth Foundations 2025</A> (tge-labs on Source Cooperative) ·{" "}
        <A href="https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/">NOAA HRRR 48 h</A> via{" "}
        <A href="https://dynamical.org">dynamical.org</A> ({s.backend === "icechunk" ? "Icechunk" : s.backend === "zarr-v3" ? "Zarr v3" : "opening"}) ·{" "}
        <A href="https://www.frames.gov/catalog/27085">Fosberg 1978</A> · model taught by Annual NLCD 2024 and Esri / Impact Observatory LULC 2024
      </p>

      {loading ? (
        <div className="field" style={{ marginTop: 8 }}>
          <div className="bar"><span style={{ width: `${Math.round(loading.fraction * 100)}%` }} /></div>
          <p className="muted small">{loading.label}. Everything is read straight from the cloud stores; nothing is precomputed but the model weights.</p>
        </div>
      ) : null}

      <Fold k="fuel" label="The ground" defaultOpen tag={p.fuel === "lit" ? "lit by FFWI" : p.fuel === "cover" ? "land cover" : "off"}>
        <div className="row">
          <Seg
            value={p.fuel}
            options={[
              { id: "lit", label: "Lit by FFWI" },
              // the land cover as classes is not shown; the picture is the ground lit by the weather
              // { id: "cover", label: "Land cover" },
              { id: "off", label: "Off" },
            ]}
            onChange={p.onFuelChange}
          />
          <Help label="The ground">
            Every pixel's 64-band AlphaEarth embedding goes through a linear head (fit offline against the 2024 land cover maps,
            held-out accuracy 0.87 to 0.92 for Esri, 0.64 to 0.92 for NLCD) to a class. Lit: the hour's FFWI times the class's
            fuel weight, hidden below the hide line, viridis from there to the ramp top. Zoomed out the embeddings come from
            the COG overviews (each overview pixel is the centroid of its children; one byte range per file and
            level, from an index built once), zoomed in from the 10 m mosaic.
          </Help>
        </div>
        <p className="muted small">Reading: {reader}{s.aefReading > 0 ? ` · ${s.aefReading} tiles in flight` : ""}</p>
        <div className="row">
          <Seg value={p.teacher} options={[{ id: "nlcd", label: "NLCD (16 classes)" }, { id: "esri", label: "Esri (9 classes)" }]} onChange={p.onTeacherChange} />
        </div>
        {p.fuel === "lit" ? (
          <>
            <div className="ramp" role="img" aria-label="Viridis ramp" style={{ backgroundImage: rampGradient() }} />
            <p className="muted small">FFWI x fuel weight, {p.hideBelow} (invisible) to {p.rescaleMax} (ramp top){p.tauHours > 0 ? ` · ${p.tauHours} h fuel lag` : ""}</p>
          </>
        ) : null}
        <Field label={`Opacity ${Math.round(p.fuelOpacity * 100)}%`}>
          <Slider min={0.05} max={1} step={0.05} value={p.fuelOpacity} onChange={p.onFuelOpacityChange} label="Ground opacity" onReset={() => p.onFuelOpacityChange(0.9)} />
        </Field>
        <Field
          label={
            <span className="row">
              <span>{p.glowHours === 0 ? "Afterglow: off (this hour only)" : `Afterglow: the brightest of the last ${p.glowHours} h, fading by fuel`}</span>
              <Help label="Afterglow">
                Each pixel shows the brightest hour behind it, faded by exp(-age / hold), with a hold per land cover: grass,
                pasture and shrub let a windy hour go in about 3 to 6 h, timber holds it about a day, built ground and water hold
                nothing. So the ground carries what the window has done to it, and timber stays lit after the grass goes dark.
              </Help>
            </span>
          }
        >
          <Slider min={0} max={48} step={1} value={p.glowHours} onChange={p.onGlowHoursChange} label="Afterglow hours" onReset={() => p.onGlowHoursChange(24)} />
        </Field>
        <Field label={p.grain === 0 ? "Grain: off (flat fill)" : `Grain: ${p.grain} cells across a tile`}>
          <Slider min={0} max={128} step={8} value={p.grain} onChange={p.onGrainChange} label="Grain" onReset={() => p.onGrainChange(0)} />
        </Field>
        <div className="row">
          <Check checked={p.light} onChange={p.onLightChange} label="Draw as light (adds up over the map)" />
          <Check checked={p.relief} onChange={p.onReliefChange} label="Relief under the ground" />
        </div>
        {/* class legend with fuel weights, not shown while the classes themselves are not drawn
        <div className="legend">
          {head.classes.map((code, i) => (
            <div key={code} className="legend-row">
              <span className="sw" style={{ background: head.colors[i] }} />
              <span>{head.names[i]}</span>
              <span className="muted">{(FUEL[p.teacher][code] ?? 0).toFixed(2)}</span>
            </div>
          ))}
        </div>
        <p className="muted small">Right column: the fuel weight each class carries (first cut, to be tuned).</p>
        */}
      </Fold>

      <Fold k="time" label="Forecast hour" defaultOpen tag={`+${hours} h · ${validStr}`}>
        <Field label="Forecast run (UTC)">
          <div className="row">
            <input
              type="date"
              min={isoDateString(INIT_TIME_ORIGIN)}
              max={p.initTimeCount > 0 ? isoDateString(dateFromInitTimeIdx(p.initTimeCount - 1)) : undefined}
              value={isoDateString(dateFromInitTimeIdx(p.initTimeIdx))}
              disabled={p.initTimeCount === 0}
              onChange={(e) => p.onInitTimeIdxChange(initTimeIdxFromDate(new Date(`${e.target.value}T12:00:00Z`), Math.max(0, p.initTimeCount - 1)))}
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
              <Help label="Forecast hour">49 hourly steps, +0 to +48 h, playing on open. Space toggles play. The run opens on the Southern Plains outbreak (2025-03-14, 12Z run; the wind peaks that afternoon from Texas to Kansas); pick any date, or jump to the latest run.</Help>
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

      <Fold k="field" label="The weather" defaultOpen tag={fieldOptions.find((o) => o.id === field)?.label}>
        <Field label={p.contourStep === 0 ? "Contours: off" : `Contours of the FFWI field every ${p.contourStep}, from the hide line up`}>
          <Slider min={0} max={50} step={5} value={p.contourStep} onChange={p.onContourStepChange} label="Contour step" onReset={() => p.onContourStepChange(0)} />
        </Field>
        <div className="row">
          <span className="muted small">Draw the 3 km field itself (puts the ground off; one picture at a time):</span>
          <Help label="FFWI">
            Fosberg 1978. Temperature and humidity set how dry fine fuel is; wind does the escalating; bone-dry fuel in a 30 mph
            wind = 100. Sustained 10 m wind is the standard input. 3 km HRRR pixels, reprojected on the GPU.
          </Help>
        </div>
        <Seg value={field} options={fieldOptions} onChange={setField} />
        {p.showRaster ? (
          <>
            <div className="row" style={{ flexWrap: "nowrap" }}>
              <select value={p.colormapId} onChange={(e) => p.onColormapChange(e.target.value as ColormapChoiceId)} style={{ flex: "0 0 auto" }}>
                {COLORMAP_CHOICES.map((c) => (<option key={c.id} value={c.id}>{c.label}</option>))}
              </select>
              <Ramp spriteUrl={colormapsPngUrl} rowCount={COLORMAP_ROW_COUNT} rowIndex={COLORMAP_INDEX[p.colormapId]!} label={fieldRampLabel} />
            </div>
            <p className="muted small">{fieldRampLabel}</p>
            <Field label={`Opacity ${Math.round(p.rasterOpacity * 100)}%`}>
              <Slider min={0} max={1} step={0.01} value={p.rasterOpacity} onChange={p.onRasterOpacityChange} label="Field opacity" onReset={() => p.onRasterOpacityChange(0.7)} />
            </Field>
          </>
        ) : null}
        <Field label={p.hideBelow > 0 ? `Hide below ${p.hideBelow} (the ground and the field both; the map shows through)` : "Showing every pixel"}>
          <Slider min={0} max={80} step={1} value={p.hideBelow} onChange={p.onHideBelowChange} label="Hide below" onReset={() => p.onHideBelowChange(20)} />
        </Field>
        <Field label={`Ramp top (fully pushing): ${p.rescaleMax}`}>
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
                instantaneous. The ground, the field and the tooltip all use the lagged value.
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
        <Check checked={p.showRaster} onChange={p.onShowRasterChange} label="Show the field" />
      </Fold>

      {s.error ? <p className="warn small" style={{ marginTop: 8 }}>{s.error}</p> : null}

      <div className="pn-foot">
        <span>Built with <A href="https://github.com/developmentseed/deck.gl-raster">deck.gl-raster</A> by <A href="https://developmentseed.org">Development Seed</A></span>
        <span>© <A href="https://carto.com/attributions">CARTO</A>, © <A href="https://www.openstreetmap.org/copyright">OpenStreetMap</A> contributors</span>
      </div>
    </div>
  );
}
