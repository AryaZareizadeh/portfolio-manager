// Run with: node tests/test_kronos_ui.cjs
// Exercise the dashboard's form and monitor logic without browser dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const fields = new Map(), storage = new Map();
function field(id) {
  if (!fields.has(id)) fields.set(id, {
    id, value: "", disabled: false, hidden: false, textContent: "", dataset: {}, style: {},
    firstChild: { textContent: "" }, selectedOptions: [{ text: "Fixed rate" }],
    classList: { toggle() {} }, setAttribute() {}, replaceChildren() {}, append() {},
    querySelector() { return field(id + "-label"); }, querySelectorAll() { return []; },
    closest() { return field(id + "-parent"); }, contains() { return true; },
  });
  if (!Object.getOwnPropertyDescriptor(fields.get(id), "value").get) {
    let value = String(fields.get(id).value);
    Object.defineProperty(fields.get(id), "value", { get: () => value, set: (next) => { value = String(next); } });
  }
  return fields.get(id);
}
const coinOrder = [];
const coins = [ { value: "BTC", checked: true }, { value: "ETH", checked: false } ];
for (const coin of coins) coin.closest = () => ({ coin });
field("coin-picks").querySelectorAll = () => coins;
field("coin-picks").append = (label) => coinOrder.push(label.coin.value);
field("training-form").querySelectorAll = () => [...fields.values()];
const context = vm.createContext({
  document: { getElementById: field, querySelector: field,
    querySelectorAll: (selector) => selector.includes(":checked") ? coins.filter((coin) => coin.checked) : coins,
    createElement: (tag) => field("new-" + tag) },
  localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
  Option: function(text, value) { this.text = text; this.value = value; },
  console, setTimeout, clearTimeout,
});
let source = fs.readFileSync(path.join(__dirname, "../crypto_forecast/web/app.js"), "utf8");
source = source.slice(0, source.indexOf('  $("training-form").addEventListener("submit"'));
source += `
  renderChart = (history) => { state.testChartHistory = history; };
  renderLogs = () => {};
  globalThis.ui = { state, applyDefaults, changeModel, syncModelControls, syncSchedulers,
    useTokenizerSettings, configFromForm, saveFormSettings, restoreFormSettings,
    applyTokenizerBootstrap, renderMonitor, setCoinSelection, organizeCoinChoices, downloadCoinView };
})();`;
vm.runInContext(source, context);
const ui = context.ui;
for (const [id, value] of Object.entries({ model_type: "gru", epochs: 60, batch_size: 128,
  learning_rate: 0.001, lookback: 60, train_fraction: 0.7, validation_fraction: 0.15,
  kronos_timeframe: "4h", kronos_prediction_length: 24, kronos_training_stage: "tokenizer",
  kronos_tokenizer_epochs: 5, kronos_tokenizer_patience: 2, kronos_tokenizer_learning_rate: 0.0002,
  lr_scheduler: "warmup_cosine", kronos_tokenizer_lr_scheduler: "warmup_cosine",
  lr_min_ratio: 0.1, kronos_tokenizer_lr_min_ratio: 0.1,
  lr_warmup_fraction: 0.05, kronos_tokenizer_lr_warmup_fraction: 0.05, start: "2021-01-01" })) field(id).value = String(value);
ui.applyDefaults({ model_type: "gru", kronos_training_stage: "sequential", kronos_tokenizer_batch_size: null });
assert.equal(field("kronos_training_stage").value, "tokenizer");
assert.equal(field("kronos_tokenizer_batch_size").value, "36");
ui.applyTokenizerBootstrap({ tokenizers: [] });
field("model_type").value = "kronos-small";
ui.changeModel();
assert.equal(field("batch_size").value, "16");
assert.equal(field("batch_size").disabled, true);
assert.equal(field("lr_scheduler").disabled, true);
assert.equal(field("kronos_tokenizer_lr_scheduler").disabled, false);
field("batch_size").value = "19";
field("kronos_training_stage").value = "predictor";
ui.syncModelControls();
assert.equal(field("batch_size").value, "19");
assert.equal(field("batch_size").disabled, false);
assert.equal(field("kronos_tokenizer_batch_size").disabled, true);
assert.equal(field("kronos_tokenizer_lr_scheduler").disabled, true);
ui.state.tokenizers = [{ id: "train-saved", name: "Saved tokenizer", complete: false,
  config: { coins: ["ETH"], start: "2022-01-01", end: "2024-12-01", lookback: 90,
    kronos_timeframe: "1h", kronos_prediction_length: 24, train_fraction: 0.65,
    validation_fraction: 0.2, kronos_train_stride: 8, kronos_eval_stride: 12 } }];
field("kronos_tokenizer_run_id").value = "train-saved";
ui.useTokenizerSettings();
assert.equal(field("start").value, "2022-01-01");
assert.equal(field("end").value, "2024-12-01");
assert.equal(coins[0].checked, false);
assert.equal(coins[1].checked, true);
assert.equal(field("batch_size").value, "19");
assert.equal(field("epochs").value, "5");
assert.equal(field("learning_rate").value, "0.00001");
assert.match(field("saved-tokenizer-note").textContent, /training stopped/);
const config = ui.configFromForm();
assert.equal(config.kronos_tokenizer_run_id, "train-saved");
assert.equal(config.kronos_finetune_tokenizer, false);
assert.equal(config.batch_size, 19);
assert.equal(config.kronos_tokenizer_batch_size, 36);
field("train_fraction").value = "0.9";
field("validation_fraction").value = "0.9";
const downloadConfig = ui.configFromForm("download");
assert.deepEqual(Object.keys(downloadConfig).sort(), ["coins", "end", "start"]);
assert.equal(downloadConfig.coins[0], "ETH");
assert.equal(downloadConfig.start, "2022-01-01");
field("train_fraction").value = "0.65";
field("validation_fraction").value = "0.2";
ui.saveFormSettings();
field("batch_size").value = "1";
ui.restoreFormSettings();
assert.equal(field("batch_size").value, "19");
ui.applyTokenizerBootstrap({});
ui.syncModelControls();
assert.equal(field("train-button").disabled, true);
assert.match(field("kronos-stage-note").textContent, /Restart the dashboard server/);
ui.state.selectedJob = "finished-tokenizer";
const history = [{ epoch: 1, train_loss: 1.2, validation_loss: 0.3 }];
ui.state.jobs = [{ id: "finished-tokenizer", status: "completed", stage: "completed",
  config: { kronos_training_stage: "tokenizer", kronos_tokenizer_epochs: 5 },
  tokenizer_history: history, history: [], progress: { best_epoch: 1 } }];
field("monitor-component").value = "auto";
ui.renderMonitor();
assert.equal(ui.state.testChartHistory, history);
assert.match(field("monitor-caption").textContent, /reconstruction/);
assert.match(field("progress-text").textContent, /available for predictor/);
ui.applyDefaults({ model_type: "kronos-small", kronos_training_stage: "sequential", kronos_finetune_tokenizer: false, batch_size: 7 });
assert.equal(field("kronos_training_stage").value, "predictor");
assert.equal(field("batch_size").value, "7");
field("kronos_training_stage").value = "sequential";
ui.syncModelControls();
assert.equal(field("batch_size").disabled, false);
assert.equal(field("kronos_tokenizer_batch_size").disabled, false);
assert.equal(field("lr_scheduler").disabled, false);
assert.equal(field("kronos_tokenizer_lr_scheduler").disabled, false);
assert.equal(ui.configFromForm().kronos_finetune_tokenizer, true);
assert.equal(ui.configFromForm().kronos_tokenizer_run_id, undefined);
ui.setCoinSelection(true);
assert.ok(coins.every((coin) => coin.checked));
assert.equal(field("coin-selection-count").textContent, "2 of 2 selected");
assert.equal(field("select-all-coins").disabled, true);
assert.equal(field("clear-coins").disabled, false);
assert.deepEqual(JSON.parse(storage.get("crypto-forecast.coin-choices.v1")).selected, ["BTC", "ETH"]);
ui.setCoinSelection(false);
assert.ok(coins.every((coin) => !coin.checked));
assert.equal(field("coin-selection-count").textContent, "0 of 2 selected");
assert.equal(field("select-all-coins").disabled, false);
assert.equal(field("clear-coins").disabled, true);
coins.reverse(); coinOrder.length = 0;
ui.organizeCoinChoices();
assert.deepEqual(coinOrder, ["BTC", "ETH"]);
const downloading = { operation: "download", status: "running", config: { coins: ["BTC", "ETH"] },
  progress: {coin_progress: { BTC: { status: "running", phase: "monthly_archives", completed: 1, total: 4 },
    ETH: {status: "queued", phase: "queued"} } } };
assert.equal(ui.downloadCoinView(downloading, "BTC").percent, 25);
assert.match(ui.downloadCoinView(downloading, "BTC").detail, /1 of 4/);
assert.equal(ui.downloadCoinView(downloading, "ETH").percent, 0);
downloading.progress.coin_progress.BTC = {status: "running", phase: "checking_source", total: null};
assert.equal(ui.downloadCoinView(downloading, "BTC").percent, null);
downloading.progress.coin_progress.BTC.status = "failed";
assert.equal(ui.downloadCoinView(downloading, "BTC").status, "Failed");
assert.equal(ui.downloadCoinView(downloading, "BTC").percent, 0);
downloading.progress.coin_progress.BTC.status = "running";
downloading.status = "stopped";
assert.equal(ui.downloadCoinView(downloading, "BTC").status, "Stopped");
assert.equal(ui.downloadCoinView(downloading, "ETH").status, "Not started");
downloading.status = "completed";
assert.equal(ui.downloadCoinView(downloading, "BTC").percent, 100);
assert.equal(ui.downloadCoinView(downloading, "BTC").status, "Saved");
downloading.status = "running";
downloading.progress.coin_progress.BTC = {status: "saved", rows: 192, completed: 4, total: 4};
assert.equal(ui.downloadCoinView(downloading, "BTC").status, "Saved");
downloading.status = "failed";
assert.equal(ui.downloadCoinView(downloading, "BTC").status, "Saved");
assert.equal(ui.downloadCoinView(downloading, "BTC").percent, 100);
downloading.status = "stopped";
assert.equal(ui.downloadCoinView(downloading, "BTC").status, "Saved");
console.log("Kronos UI smoke passed: stages, batches, schedulers, reuse, persistence, compatibility, completed curves.");
