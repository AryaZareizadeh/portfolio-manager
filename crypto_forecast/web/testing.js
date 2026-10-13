(() => {
  "use strict";
  const {forecastWindow, windowDescription, restoreWindow, modelChoices, pretrainedModel, modelReady} = window.KronosForecastControls;
  const $ = (id) => document.getElementById("test-" + id);
  const active = (job) => ["queued", "running", "awaiting_decision"].includes(job?.status);
  const state = {token: null, data: [], models: null, jobs: [], chart: null, view: null,
    anchor: null, selected: null, loaded: null, loadingResult: null, busy: false, version: 0};
  const utc = (value) => new Date(value).toISOString().replace("T", " ").slice(0, 16) + " UTC";
  const number = (value) => value == null ? "—" : Number(value).toLocaleString("en-US", {maximumFractionDigits: 6});
  function error(message = "") { $("error").textContent = message; $("error").hidden = !message; }
  async function api(path, value) {
    const response = await fetch(path, value === undefined ? {} : {method: "POST",
      headers: {"Content-Type": "application/json", "X-CSRF-Token": state.token}, body: JSON.stringify(value)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "The local server could not complete this request.");
    return body;
  }
  function settings() {
    return {coin: $("coin").value, timeframe: $("timeframe").value, model: $("model").value,
      lookback: Number($("lookback").value),
      horizon: forecastWindow(Number($("duration").value), $("duration-unit").value, $("timeframe").value).horizon};
  }
  function windowNote() {
    try {
      const result = forecastWindow(Number($("duration").value), $("duration-unit").value, $("timeframe").value);
      $("duration").setCustomValidity(""); $("duration-note").textContent = windowDescription(result);
    } catch (reason) {
      $("duration").setCustomValidity(reason.message); $("duration-note").textContent = reason.message;
    }
  }
  function coins() {
    const selected = $("coin").value;
    const choices = state.data.filter((entry) => entry.timeframes.includes($("timeframe").value));
    $("coin").replaceChildren(...choices.map((entry) => new Option(entry.coin, entry.coin)));
    if (choices.some((entry) => entry.coin === selected)) $("coin").value = selected;
    if (!choices.length) $("coin").append(new Option("Download history in the training tab", ""));
  }
  function actions() {
    const running = state.jobs.find(active), job = state.jobs.find((job) => job.id === state.selected);
    const ownActive = active(job), ready = modelReady(state.models, $("model").value);
    $("rerun").disabled = state.busy || !state.token || !state.chart || !state.anchor || !!running || !ready;
    $("stop").disabled = state.busy || !state.token || !ownActive;
    $("refresh").disabled = state.busy || !state.token || ownActive;
    for (const id of ["model", "coin", "timeframe", "lookback", "duration", "duration-unit", "jobs"]) $(id).disabled = state.busy || ownActive;
    const original = pretrainedModel(state.models, $("model").value);
    $("model-note").textContent = !original
      ? "Separate fine-tuned checkpoint. Choose the candle interval used to train it. Weights remain unchanged."
      : original.ready ? `Original downloaded ${original.name} weights. Model tests leave the weights unchanged.`
        : "Original weights are missing. Download them before starting a model test.";
    $("status").textContent = state.busy ? "Working…" : job?.status || (state.chart ? "Click a candle" : "Ready");
    $("status").className = "status-badge " + (job?.status || "idle");
    $("progress").textContent = job ? (job.logs?.at(-1)?.message || job.stage || job.status) + ` · ${Math.round(job.elapsed_seconds || 0)}s`
      : running ? "Another job is running. You can explore the chart; wait for that job before starting a test." : "";
  }
  function jobs() {
    const entries = state.jobs.filter((job) => job.mode === "testing"), select = $("jobs");
    select.replaceChildren(new Option(entries.length ? "Select a saved model test" : "No model tests yet", ""));
    for (const job of entries) select.append(new Option(`${job.name} · ${job.status}`, job.id));
    select.value = state.selected || ""; actions();
  }
  function draw(preserveRange = false) {
    if (!state.chart) return;
    $("empty").hidden = true;
    if (!state.view) state.view = window.KronosChart.create($("chart"), {
      readout: $("readout"), volumeTitle: $("volume-title"), forecastRegion: $("forecast-region"), lookbackRegion: $("lookback-region"),
      fit: $("fit"), focus: $("focus"), zoomIn: $("zoom-in"), zoomOut: $("zoom-out"),
    }, {compare: true, onSelect: (candle) => predict(candle.timestamp)});
    state.view.setData(state.chart, {preserveRange});
    const history = state.chart.history;
    $("description").textContent = `${state.chart.coin} · ${state.chart.timeframe} · ${history.length.toLocaleString()} completed historical candles`;
    $("range").textContent = `${utc(history[0].timestamp)} to ${utc(history.at(-1).timestamp)}. All saved history is loaded.`
      + (state.chart.synthetic ? " Synthetic demonstration data." : "")
      + (state.chart.lookback_window ? ` Lookback: ${utc(state.chart.lookback_window.start)} to ${utc(state.chart.lookback_window.end)} (${state.chart.lookback_window.count} candles).` : "");
  }
  function clearSelection() {
    state.anchor = state.selected = state.loaded = null;
    $("anchor").textContent = "Click a candle in the price pane.";
    $("result").hidden = true;
  }
  async function loadChart() {
    if (!state.token || !$("coin").value) return;
    const version = ++state.version;
    state.busy = true; clearSelection(); state.chart = null;
    // Remove the previous coin immediately so a stale chart cannot be clicked during loading.
    $("chart").style.visibility = "hidden"; $("empty").hidden = false;
    $("empty").textContent = "Loading full saved chart…"; $("forecast-region").hidden = $("lookback-region").hidden = true;
    error(); actions(); jobs();
    try {
      const result = await api("/api/testing/chart", {coin: $("coin").value, timeframe: $("timeframe").value, model: $("model").value});
      if (version !== state.version) return;
      state.chart = result; $("chart").style.visibility = ""; draw();
    } catch (reason) {
      if (version === state.version) { error(reason.message); $("empty").textContent = "Chart unavailable. Check the saved history and reload."; }
    } finally { if (version === state.version) { state.busy = false; actions(); } }
  }
  async function predict(anchor = state.anchor) {
    if (!anchor || !state.chart || state.busy || state.jobs.some(active)) return;
    windowNote();
    if (!$("form").reportValidity()) return;
    state.anchor = anchor;
    $("anchor").textContent = utc(anchor) + " · last input candle";
    state.selected = state.loaded = null; $("result").hidden = true;
    const value = settings();
    const index = state.chart.history.findIndex((candle) => candle.timestamp === anchor);
    state.chart = {...state.chart, forecast: [], selected_candle: anchor,
      lookback_window: index >= value.lookback - 1 ? {start: state.chart.history[index - value.lookback + 1].timestamp,
        end: anchor, count: value.lookback} : null}; draw(true);
    if (index < value.lookback - 1) { error(`Need ${value.lookback} candles up to the selected point; only ${index + 1} are available. Select a later candle or reduce lookback.`); jobs(); return; }
    const ready = modelReady(state.models, value.model);
    if (!ready) { error("Download the selected original Kronos weights before starting a model test."); actions(); return; }
    state.busy = true; error(); actions();
    try {
      const job = await api("/api/testing/forecast", {...value, anchor});
      state.selected = job.id; state.jobs.unshift(job);
      window.dispatchEvent(new CustomEvent("workspace-jobs", {detail: state.jobs}));
    } catch (reason) { error(reason.message); }
    finally { state.busy = false; jobs(); }
  }
  function showResult(result) {
    state.anchor = result.selected_candle;
    $("anchor").textContent = utc(state.anchor) + " · last input candle";
    state.chart = {...state.chart, forecast: result.forecast, selected_candle: result.selected_candle,
      lookback_window: {start: result.history[0].timestamp, end: result.history.at(-1).timestamp,
        count: result.history.length}}; draw(true);
    $("result").hidden = false;
    $("result-model").textContent = `${result.model_label} · ${result.device} · ${result.settings.lookback} input candles · ${result.settings.horizon} predictions`;
    $("json").href = "/api/chart/forecasts/" + encodeURIComponent(state.selected);
    const actual = new Map((result.actual || []).map((candle) => [Date.parse(candle.timestamp), candle]));
    const matched = result.forecast.filter((candle) => actual.has(Date.parse(candle.timestamp))).length;
    $("comparison-note").textContent = `Forecast starts ${utc(result.origin)}. ${matched} of ${result.forecast.length} predictions have observed candles for comparison. Missing observations appear as —. Only the ${result.settings.lookback} candles ending at the selected point were supplied to the model. ${result.amount_note}`;
    $("comparison-body").replaceChildren(...result.forecast.map((candle) => {
      const observed = actual.get(Date.parse(candle.timestamp)), row = document.createElement("tr");
      for (const value of [utc(candle.timestamp), number(candle.close), number(observed?.close), number(candle.volume), number(observed?.volume)]) {
        const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
      }
      return row;
    }));
    $("adjustment").textContent = result.adjusted_candles ? `${result.adjusted_candles} candle ranges adjusted for display. ${result.adjustment_note}` : "";
  }
  async function loadResult() {
    const job = state.jobs.find((entry) => entry.id === state.selected);
    if (job?.error) error(job.error);
    if (job?.status !== "completed" || state.loaded === job.id || state.loadingResult === job.id) return;
    const identifier = job.id, version = state.version;
    state.loadingResult = identifier;
    try {
      const result = await api("/api/chart/forecasts/" + encodeURIComponent(identifier));
      if (state.selected === identifier && version === state.version && state.chart) {
        state.loaded = identifier; showResult(result); actions();
      }
    } catch (reason) { error(reason.message); }
    finally { state.loadingResult = null; }
  }
  function bootstrap(body) {
    state.token = body.csrf_token; state.data = body.data?.coins || []; state.models = body.chart_models;
    state.jobs = body.jobs || [];
    const selected = $("model").value;
    $("model").replaceChildren(...modelChoices(state.models).map((choice) => new Option(choice.label, choice.value)));
    if ([...$("model").options].some((option) => option.value === selected)) $("model").value = selected;
    coins(); windowNote(); jobs();
    if (!document.getElementById("testing-tab").hidden && !state.chart && !state.busy) loadChart();
  }
  window.addEventListener("workspace-bootstrap", (event) => bootstrap(event.detail));
  window.addEventListener("workspace-jobs", (event) => { state.jobs = event.detail; jobs(); loadResult(); });
  window.addEventListener("workspace-tab", (event) => {
    if (event.detail === "testing" && !state.chart && !state.busy) loadChart();
  });
  $("form").addEventListener("submit", (event) => { event.preventDefault(); predict(); });
  $("refresh").addEventListener("click", async () => {
    try { await window.modelWorkspace.refresh(); await loadChart(); }
    catch (reason) { error(reason.message); }
  });
  $("coin").addEventListener("change", loadChart);
  $("model").addEventListener("change", loadChart);
  $("timeframe").addEventListener("change", () => { coins(); windowNote(); loadChart(); });
  $("duration").addEventListener("input", windowNote);
  $("duration-unit").addEventListener("change", windowNote);
  $("stop").addEventListener("click", async () => {
    state.busy = true; actions();
    try {
      const job = await api("/api/jobs/" + encodeURIComponent(state.selected) + "/stop", {});
      state.jobs = state.jobs.map((entry) => entry.id === job.id ? job : entry);
    } catch (reason) { error(reason.message); }
    finally { state.busy = false; jobs(); }
  });
  $("jobs").addEventListener("change", async () => {
    const identifier = $("jobs").value, job = state.jobs.find((entry) => entry.id === identifier);
    if (!job) { clearSelection(); if (state.chart) { state.chart.forecast = []; delete state.chart.selected_candle; delete state.chart.lookback_window; draw(true); } jobs(); return; }
    for (const key of ["model", "timeframe", "lookback"]) $(key).value = job.config[key];
    const restored = restoreWindow(job.config.horizon, job.config.timeframe);
    $("duration").value = restored.duration; $("duration-unit").value = restored.unit; windowNote();
    coins(); $("coin").value = job.config.coin;
    await loadChart();
    state.selected = identifier; state.anchor = job.anchor;
    $("anchor").textContent = utc(job.anchor) + " · last input candle";
    jobs(); await loadResult();
  });
  if (window.modelWorkspace) bootstrap({...window.modelWorkspace.bootstrap, jobs: window.modelWorkspace.jobs});
})();
