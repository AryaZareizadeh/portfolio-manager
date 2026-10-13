(() => {
  "use strict";
  function forecastWindow(duration, unit, timeframe) {
    const intervals = {"15m": 0.25, "1h": 1, "4h": 4, "1d": 24};
    const hoursPerCandle = intervals[timeframe];
    if (!Number.isFinite(duration) || duration <= 0 || !hoursPerCandle || !["hours", "days", "candles"].includes(unit)) {
      throw new Error("Choose a positive forecast duration and a supported interval.");
    }
    const requestedCandles = unit === "candles" ? duration : duration * (unit === "days" ? 24 : 1) / hoursPerCandle;
    const horizon = Math.ceil(requestedCandles - 1e-10);
    if (horizon < 1 || horizon > 512) {
      throw new Error(`This window needs ${horizon} future candles. Choose at most 512 candles (${512 * hoursPerCandle / 24} days at this interval).`);
    }
    return {horizon, hours: horizon * hoursPerCandle, rounded: Math.abs(horizon - requestedCandles) > 1e-9};
  }
  function windowDescription(result) {
    const duration = result.hours >= 24 ? result.hours / 24 : result.hours;
    const unit = result.hours >= 24 ? "day" : "hour";
    return `${result.horizon} future candle${result.horizon === 1 ? "" : "s"} · ${duration} ${unit}${duration === 1 ? "" : "s"}${result.rounded ? " · Rounded up to complete candles." : "."}`;
  }
  function restoreWindow(horizon, timeframe) {
    const hours = horizon * ({"15m": 0.25, "1h": 1, "4h": 4, "1d": 24})[timeframe];
    return hours >= 24 && hours % 24 === 0 ? {duration: hours / 24, unit: "days"} : {duration: hours, unit: "hours"};
  }
  function modelChoices(models) {
    return [...(models?.pretrained || [{id: "original", name: "Kronos-small"}]).map((model) => ({label: model.name + " · original weights", value: model.id})),
      ...(models?.runs || []).map((run) => ({label: "Fine-tuned · " + run.name, value: run.id}))];
  }
  function pretrainedModel(models, identifier) {
    return models?.pretrained?.find((model) => model.id === identifier)
      || (identifier === "original" ? {name: "Kronos-small", ready: !!models?.original_ready} : null);
  }
  function modelReady(models, identifier) {
    const original = pretrainedModel(models, identifier);
    return original ? original.ready : identifier !== "original-base";
  }
  // Export the conversion for boundary tests without starting browser polling.
  const controls = {forecastWindow, windowDescription, restoreWindow, modelChoices, pretrainedModel, modelReady};
  if (typeof module !== "undefined" && module.exports) { module.exports = Object.assign(forecastWindow, controls); return; }
  window.KronosForecastControls = controls;
  const $ = (id) => document.getElementById(id);
  const active = (job) => ["queued", "running", "awaiting_decision"].includes(job?.status);
  const state = {token: null, data: [], models: null, jobs: [], selected: null, loaded: null,
    chart: null, result: null, busy: false, online: false};
  const tabs = ["chart", "testing", "training"];
  function selectTab(name) {
    for (const tab of tabs) {
      $(tab + "-tab").hidden = tab !== name;
      $(tab + "-tab-button").setAttribute("aria-selected", String(tab === name));
      $(tab + "-tab-button").tabIndex = tab === name ? 0 : -1;
    }
    window.dispatchEvent(new Event("resize"));
    window.dispatchEvent(new CustomEvent("workspace-tab", {detail: name}));
  }
  tabs.forEach((name, index) => {
    const button = $(name + "-tab-button");
    button.addEventListener("click", () => selectTab(name));
    button.addEventListener("keydown", (event) => {
      if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
          : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        selectTab(tabs[next]); $(tabs[next] + "-tab-button").focus();
      }
    });
  });
  function error(message = "") { $("chart-error").textContent = message; $("chart-error").hidden = !message; }
  async function api(path, value) {
    const response = await fetch(path, value === undefined ? {} : {method: "POST",
      headers: {"Content-Type": "application/json", "X-CSRF-Token": state.token}, body: JSON.stringify(value)});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "The local server could not complete this request.");
    return body;
  }
  function settings() {
    const window = forecastWindow(Number($("chart-duration").value), $("chart-duration-unit").value, $("chart-timeframe").value);
    const value = {source: "saved", coin: $("chart-coin").value,
      timeframe: $("chart-timeframe").value, model: $("chart-model").value,
      lookback: Number($("chart-lookback").value), horizon: window.horizon,
      sample_count: Number($("chart-samples").value), seed: Number($("chart-seed").value),
      temperature: Number($("chart-temperature").value), top_p: Number($("chart-top-p").value),
      device: $("chart-device").value, end: $("chart-end").value ? $("chart-end").value + "Z" : null};
    return value;
  }
  function windowNote() {
    try {
      const result = forecastWindow(Number($("chart-duration").value), $("chart-duration-unit").value, $("chart-timeframe").value);
      $("chart-duration").setCustomValidity("");
      $("chart-duration-note").textContent = windowDescription(result);
    } catch (reason) {
      $("chart-duration").setCustomValidity(reason.message);
      $("chart-duration-note").textContent = reason.message;
    }
  }
  function actions() {
    const running = state.jobs.find(active);
    const selected = state.jobs.find((job) => job.id === state.selected);
    const ready = modelReady(state.models, $("chart-model").value);
    $("chart-forecast-button").disabled = state.busy || !state.online || !!running || !ready;
    $("chart-preview-button").disabled = state.busy || !state.online;
    $("chart-stop").disabled = state.busy || !state.online || !active(selected);
    if (running && running.operation !== "forecast") $("chart-job-progress").textContent = "A training or download job is running. Wait for it to finish before forecasting.";
  }
  function coins() {
    const selected = $("chart-coin").value;
    const choices = state.data.filter((entry) => entry.timeframes.includes($("chart-timeframe").value));
    $("chart-coin").replaceChildren(...choices.map((entry) => new Option(entry.coin, entry.coin)));
    if (choices.some((entry) => entry.coin === selected)) $("chart-coin").value = selected;
    if (!choices.length) $("chart-coin").append(new Option("Download coin history in the training tab", ""));
  }
  async function bootstrap() {
    const body = await api("/api/bootstrap");
    state.token = body.csrf_token; state.data = body.data?.coins || [];
    state.models = body.chart_models; state.jobs = body.jobs || []; state.online = true;
    const selected = $("chart-model").value;
    $("chart-model").replaceChildren(...modelChoices(state.models).map((choice) => new Option(choice.label, choice.value)));
    if ([...$("chart-model").options].some((option) => option.value === selected)) $("chart-model").value = selected;
    modelNote(); coins(); windowNote(); renderJobs(); actions();
    window.modelWorkspace = {bootstrap: body, jobs: state.jobs, refresh: bootstrap};
    window.dispatchEvent(new CustomEvent("workspace-bootstrap", {detail: body}));
  }
  function modelNote() {
    const original = pretrainedModel(state.models, $("chart-model").value);
    $("chart-model-note").textContent = !original
      ? "Using a separate fine-tuned checkpoint. Match its trained candle interval. Forecasting leaves all weight files unchanged."
      : original.ready ? `Original downloaded ${original.name} weights. Fine-tuning is optional; forecasts leave the weights unchanged.`
        : "Original weights are missing. Run python -m crypto_forecast kronos-setup in the project terminal.";
  }
  function number(value) { return Number(value).toLocaleString(undefined, {maximumFractionDigits: 6}); }
  function utc(value) { return new Date(value).toISOString().replace("T", " ").slice(0, 16); }
  let chartView = null;
  function draw(chart) {
    $("candle-empty").hidden = !!chart;
    if (!chart) return;
    if (!chartView) chartView = window.KronosChart.create($("candle-chart"), {
      readout: $("candle-readout"), volumeTitle: $("candle-volume-title"), forecastRegion: $("candle-forecast-region"),
      fit: $("candle-fit"), focus: $("candle-focus"), zoomIn: $("candle-zoom-in"), zoomOut: $("candle-zoom-out"),
    });
    chartView.setData(chart);
    const history = chart.history, future = chart.forecast || [];
    $("chart-description").textContent = `${chart.coin} · ${chart.timeframe} · ${history.length} history candles${future.length ? " · " + future.length + " future candles" : ""}`;
    $("chart-window-note").textContent = `UTC open times · Last completed candle closes ${utc(chart.origin)} UTC · Forecast window ends ${utc(chart.forecast_end)} UTC.`;
    const warnings = [];
    if (chart.historical) warnings.push("This is a historical chart window. Its forecast begins after the selected last candle, rather than at the current time.");
    if (chart.synthetic) warnings.push("This chart uses synthetic demonstration data.");
    if (chart.incomplete_dropped) warnings.push(`${chart.incomplete_dropped} candles after the cutoff or still forming were excluded.`);
    warnings.push(chart.amount_note);
    $("chart-data-note").textContent = warnings.join(" "); $("chart-data-note").hidden = false;
  }
  function renderResult(result) {
    state.result = result; state.chart = result; draw(result); $("candle-result").hidden = false;
    $("chart-result-model").textContent = result.model_label + ` · ${result.device} · seed ${result.settings.seed}`;
    $("chart-adjustment").textContent = result.adjusted_candles ? `${result.adjusted_candles} candle ranges adjusted for display/export. ${result.adjustment_note}` : "";
    $("candle-forecast-body").replaceChildren(...result.forecast.map((candle) => {
      const row = document.createElement("tr");
      for (const value of [utc(candle.timestamp), ...["open", "high", "low", "close", "volume"].map((key) => number(candle[key]))]) {
        const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
      }
      return row;
    }));
    $("chart-json").href = "/api/chart/forecasts/" + encodeURIComponent(state.selected);
  }
  function renderJobs() {
    const jobs = state.jobs.filter((job) => job.operation === "forecast" && job.mode !== "testing"), select = $("chart-job-select");
    select.replaceChildren(new Option(jobs.length ? "Select a saved forecast" : "No candle forecasts yet", ""));
    for (const job of jobs) select.append(new Option(`${job.name} · ${job.status}`, job.id));
    select.value = state.selected || "";
    const job = jobs.find((entry) => entry.id === state.selected);
    $("chart-status").textContent = state.busy ? "Working…" : job?.status || "Ready";
    $("chart-status").className = "status-badge " + (job?.status || "idle");
    $("chart-job-progress").textContent = job ? (job.logs?.at(-1)?.message || job.stage) + ` · ${Math.round(job.elapsed_seconds)}s` : "";
    if (job?.error) error(job.error);
    actions();
  }
  async function loadSelected() {
    const job = state.jobs.find((entry) => entry.id === state.selected);
    if (job?.status === "completed" && state.loaded !== job.id) {
      const identifier = job.id, result = await api("/api/chart/forecasts/" + encodeURIComponent(identifier));
      if (state.selected === identifier) {
        state.loaded = identifier;
        for (const [key, id] of Object.entries({lookback: "lookback", seed: "seed",
          sample_count: "samples", temperature: "temperature", top_p: "top-p", device: "device", model: "model", timeframe: "timeframe"})) {
          $("chart-" + id).value = job.config[key];
        }
        coins(); $("chart-coin").value = job.config.coin;
        const restored = restoreWindow(job.config.horizon, job.config.timeframe);
        $("chart-duration-unit").value = restored.unit;
        $("chart-duration").value = restored.duration; windowNote();
        $("chart-end").value = job.config.end ? new Date(job.config.end).toISOString().slice(0, 16) : "";
        modelNote(); actions(); renderResult(result);
      }
    }
  }
  function clearResult() { state.result = null; state.loaded = null; $("candle-result").hidden = true; }
  async function submit(forecast) {
    if (!$("chart-form").reportValidity()) return;
    state.busy = true; error(); actions();
    try {
      const value = settings();
      const chart = await api("/api/chart/preview", value);
      state.selected = null; clearResult(); state.chart = chart; draw(chart);
      if (forecast) {
        const job = await api("/api/chart/forecast", value);
        state.selected = job.id; state.jobs.unshift(job);
      }
      renderJobs();
    } catch (reason) { error(reason.message); }
    finally { state.busy = false; renderJobs(); }
  }
  $("chart-form").addEventListener("submit", (event) => { event.preventDefault(); submit(true); });
  $("chart-preview-button").addEventListener("click", () => submit(false));
  $("chart-timeframe").addEventListener("change", () => { coins(); windowNote(); });
  $("chart-duration").addEventListener("input", windowNote);
  $("chart-duration-unit").addEventListener("change", windowNote);
  $("chart-model").addEventListener("change", () => { modelNote(); actions(); });
  $("chart-refresh").addEventListener("click", () => bootstrap().catch((reason) => error(reason.message)));
  $("chart-job-select").addEventListener("change", () => {
    state.selected = $("chart-job-select").value || null; clearResult(); error(); renderJobs();
    loadSelected().catch((reason) => error(reason.message));
  });
  $("chart-stop").addEventListener("click", async () => {
    state.busy = true; actions();
    try { const job = await api("/api/jobs/" + encodeURIComponent(state.selected) + "/stop", {});
      state.jobs = state.jobs.map((entry) => entry.id === job.id ? job : entry); }
    catch (reason) { error(reason.message); }
    finally { state.busy = false; renderJobs(); }
  });
  $("chart-download").addEventListener("click", () => {
    if (!state.result) return;
    const columns = ["timestamp", "open", "high", "low", "close", "volume"];
    const csv = [columns.join(","), ...state.result.forecast.map((row) => columns.map((column) => row[column]).join(","))].join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], {type: "text/csv;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url;
    link.download = `${state.result.coin}-${state.result.timeframe}-kronos-forecast.csv`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  async function poll() {
    try {
      if (!state.token || !state.online) await bootstrap();
      state.jobs = (await api("/api/jobs")).jobs || []; state.online = true;
      window.modelWorkspace.jobs = state.jobs;
      window.dispatchEvent(new CustomEvent("workspace-jobs", {detail: state.jobs}));
      renderJobs(); await loadSelected();
    } catch (reason) { state.online = false; error(reason.message); actions(); }
    finally { setTimeout(poll, state.jobs.some(active) ? 1500 : 3000); }
  }
  bootstrap().then(() => {
    const job = state.jobs.find((entry) => entry.operation === "forecast" && entry.mode !== "testing" && active(entry));
    state.selected = job?.id || null; renderJobs();
  }).catch((reason) => error(reason.message)).finally(() => setTimeout(poll, 1500));
})();
