const test = require("node:test");
const assert = require("node:assert/strict");
const forecastWindow = require("../crypto_forecast/web/chart.js");

test("saved forecast lengths restore to equivalent duration controls for either tab", () => {
  for (const timeframe of ["15m", "1h", "4h", "1d"]) {
    for (let horizon = 1; horizon <= 512; horizon++) {
      const restored = forecastWindow.restoreWindow(horizon, timeframe);
      assert.equal(forecastWindow(restored.duration, restored.unit, timeframe).horizon, horizon);
    }
  }
});

test("both tabs describe converted and rounded windows consistently", () => {
  assert.equal(forecastWindow.windowDescription(forecastWindow(24, "hours", "4h")), "6 future candles · 1 day.");
  assert.equal(forecastWindow.windowDescription(forecastWindow(6, "hours", "4h")), "2 future candles · 8 hours · Rounded up to complete candles.");
});

test("selected time windows produce the right number of future candles", () => {
  assert.deepEqual(forecastWindow(24, "hours", "4h"), {horizon: 6, hours: 24, rounded: false});
  assert.deepEqual(forecastWindow(7, "days", "4h"), {horizon: 42, hours: 168, rounded: false});
  assert.deepEqual(forecastWindow(24, "hours", "15m"), {horizon: 96, hours: 24, rounded: false});
  assert.deepEqual(forecastWindow(7, "days", "1d"), {horizon: 7, hours: 168, rounded: false});
});

test("partial candles are rounded up and disclosed", () => {
  assert.deepEqual(forecastWindow(6, "hours", "4h"), {horizon: 2, hours: 8, rounded: true});
  assert.deepEqual(forecastWindow(1, "hours", "1d"), {horizon: 1, hours: 24, rounded: true});
});

test("reject windows beyond the decoder limit without silently truncating", () => {
  assert.equal(forecastWindow(512, "candles", "4h").horizon, 512);
  assert.equal(forecastWindow(128, "hours", "15m").horizon, 512);
  assert.throws(() => forecastWindow(513, "candles", "4h"), /at most 512/);
  assert.throws(() => forecastWindow(7, "days", "15m"), /672 future candles/);
});

test("reject nonpositive, nonfinite and unknown durations", () => {
  for (const duration of [0, -1, NaN, Infinity]) assert.throws(() => forecastWindow(duration, "hours", "4h"));
  assert.throws(() => forecastWindow(1, "months", "4h"));
  assert.throws(() => forecastWindow(1, "days", "5m"));
});
