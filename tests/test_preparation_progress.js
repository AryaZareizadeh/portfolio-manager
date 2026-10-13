const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

// Exercise the actual monitor renderer with a small DOM and no chart/browser dependencies.
const app = fs.readFileSync(path.join(__dirname, "../crypto_forecast/web/app.js"), "utf8");
const renderer = app.slice(app.indexOf("  function renderMonitor()"), app.indexOf("  function renderRuns()"));
function monitor(job) {
  const elements = new Map();
  const context = {
    state: {jobs: [job], selectedJob: job.id},
    $: (id) => {
      if (!elements.has(id)) elements.set(id, {value: "auto", style: {}, attributes: {}, classes: {},
        setAttribute(key, value) { this.attributes[key] = value; },
        classList: {toggle(key, value) { elements.get(id).classes[key] = value; }}});
      return elements.get(id);
    },
    active: (job) => ["queued", "running", "awaiting_decision"].includes(job.status),
    waiting: (job) => job.status === "awaiting_decision",
    count: String, number: String, duration: String, rateText: String,
    showError() {}, renderDecision() {}, renderChart() {}, renderLogs() {}, updateActions() {},
  };
  vm.runInNewContext(`${renderer}\nrenderMonitor();`, context);
  return Object.fromEntries(elements);
}
const job = (progress, stage = "preparing") => ({id: "test", status: "running", operation: "train",
  stage, config: {model_type: "kronos-small", coins: ["BTC"]}, progress});

test("candle preparation displays zero, intermediate and complete percentages and current coin", () => {
  for (const completed of [0, 50, 100]) {
    const dom = monitor(job({preparation_phase: "windows", preparation_completed: completed,
      preparation_total: 100, coin: "BTC", usable_samples: 42}));
    assert.equal(dom["progress-fill"].style.width, `${completed}%`);
    assert.equal(dom["progress-percent"].textContent, `${completed}%`);
    assert.equal(dom["progress-track"].attributes["aria-valuenow"], completed);
    assert.equal(dom["progress-track"].classes.indeterminate, false);
    assert.match(dom["progress-text"].textContent, /candle windows checked · BTC · 42 usable examples/);
    assert.equal(dom["stage-value"].textContent, "Building candle examples");
  }
});

test("split and model loading stages do not retain the completed candle preparation bar", () => {
  for (const [phase, stage] of [["splitting", "preparing"], ["windows", "loading_model"]]) {
    const dom = monitor(job({preparation_phase: phase, preparation_completed: 100, preparation_total: 100}, stage));
    assert.equal(dom["progress-track"].classes.indeterminate, true);
    assert.equal(dom["progress-percent"].textContent, "");
    assert.equal(dom["progress-fill"].style.width, "0%");
    assert.match(dom["progress-text"].textContent, phase === "splitting" ? /Splitting/ : /Loading pretrained/);
  }
});

test("tokenization shows the current partition and resets progress for validation", () => {
  for (const [partition, completed] of [["training", 50], ["validation", 0]]) {
    const dom = monitor(job({preparation_phase: "tokenizing", preparation_partition: partition,
      preparation_completed: completed, preparation_total: 100, total_batches: 10, batch: 10}));
    assert.equal(dom["progress-percent"].textContent, `${completed}%`);
    assert.match(dom["progress-text"].textContent, new RegExp(`Tokenizing ${partition} samples`));
  }
});

test("training continues to show batch progress even with retained preparation fields", () => {
  const dom = monitor(job({preparation_phase: "windows", preparation_completed: 100,
    preparation_total: 100, total_batches: 20, batch: 5}, "training"));
  assert.equal(dom["progress-percent"].textContent, "25%");
  assert.equal(dom["progress-text"].textContent, "Batch 5 of 20");
});

test("selected-window preparation reports progress and CPU workers", () => {
  const dom = monitor(job({preparation_phase: "materializing", preparation_completed: 25,
    preparation_total: 100, preparation_workers: 4}));
  assert.equal(dom["progress-percent"].textContent, "25%");
  assert.equal(dom["progress-track"].classes.indeterminate, false);
  assert.match(dom["progress-text"].textContent, /25 of 100/);
  assert.match(dom["stage-value"].textContent, /4 CPU workers/);
});
