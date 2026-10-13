const test = require("node:test");
const assert = require("node:assert/strict");
const {seriesData, priceFormat, formatTime, tickTime} = require("../crypto_forecast/web/chart_view.js");

test("price and volume share timestamps, preserve values and distinguish forecasts", () => {
  const chart = {history: [
    {timestamp: "2026-10-10T04:00:00Z", open: 2, high: 4, low: 1, close: 3, volume: 500},
    {timestamp: "2026-10-10T08:00:00Z", open: 3, high: 4, low: 1, close: 2, volume: 0},
  ], forecast: [{timestamp: "2026-10-10T12:00:00Z", open: 2, high: 4, low: 1, close: 3, volume: 125.75}]};
  const before = JSON.stringify(chart);
  const result = seriesData(chart);
  assert.deepEqual(result.prices.map((row) => row.time), result.volumes.map((row) => row.time));
  assert.deepEqual(result.volumes.map((row) => row.value), [500, 0, 125.75]);
  assert.deepEqual(result.rows.map((row) => row.forecast), [false, false, true]);
  assert.equal(result.prices[2].color, "#6775c8");
  assert.equal(result.volumes[2].color, "#6775c8b3");
  assert.notEqual(result.prices[0].color, result.prices[1].color);
  assert.equal(result.prices[2].close, chart.forecast[0].close);
  assert.equal(JSON.stringify(chart), before);
});

test("history-only preview has a full volume series and no forecast flags", () => {
  const result = seriesData({history: [{timestamp: "2026-10-10T04:00:00Z", open: 1, high: 2, low: 1, close: 2, volume: 42}]});
  assert.equal(result.volumes.length, 1);
  assert.equal(result.volumes[0].value, 42);
  assert.equal(result.rows[0].forecast, false);
});

test("price axes retain precision for small coins", () => {
  assert.equal(priceFormat([{close: 65000}]).precision, 2);
  assert.equal(priceFormat([{close: 0.015}]).precision, 4);
  assert.equal(priceFormat([{close: 0.00001234}]).precision, 7);
  assert.equal(priceFormat([]).precision, 2);
});

test("time and crosshair labels use UTC even on a different local timezone", () => {
  const time = Date.parse("2026-10-10T23:45:00Z") / 1000;
  assert.equal(formatTime(time), "2026-10-10 23:45 UTC");
  assert.equal(tickTime(time, 3), "23:45");
  assert.equal(tickTime(time, 2), "10 Oct");
  assert.equal(tickTime(time, 0), "2026");
});

test("price formatting supports a full multi-year intraday history without argument overflow", () => {
  const rows = Array.from({length: 200000}, (_, index) => ({close: index === 100000 ? 0.001 : 65000}));
  assert.equal(priceFormat(rows).precision, 5);
});
