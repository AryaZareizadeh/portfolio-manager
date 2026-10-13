(() => {
  "use strict";
  const colors = {up: "#168277", down: "#b54c5d", forecast: "#6775c8"};
  function seriesData(chart) {
    const history = chart.history || [], forecast = chart.forecast || [];
    const rows = [...history, ...forecast].map((candle, index) => ({...candle,
      time: Date.parse(candle.timestamp) / 1000, forecast: index >= history.length}));
    return {rows, prices: rows.map((row) => {
      const color = row.forecast ? colors.forecast : row.close >= row.open ? colors.up : colors.down;
      return {time: row.time, open: row.open, high: row.high, low: row.low, close: row.close,
        color, borderColor: color, wickColor: color};
    }), volumes: rows.map((row) => ({time: row.time, value: row.volume,
      color: row.forecast ? "#6775c8b3" : row.close >= row.open ? "#16827799" : "#b54c5d99"}))};
  }
  function priceFormat(rows) {
    const minimum = rows.reduce((minimum, row) => row.close > 0 ? Math.min(minimum, row.close) : minimum, Infinity);
    const precision = Number.isFinite(minimum) ? Math.min(10, Math.max(2, 2 - Math.floor(Math.log10(minimum)))) : 2;
    return {type: "price", precision, minMove: 10 ** -precision};
  }
  function formatTime(time) {
    return new Date(time * 1000).toISOString().replace("T", " ").slice(0, 16) + " UTC";
  }
  function tickTime(time, type) {
    const date = new Date(time * 1000);
    if (type === 0) return String(date.getUTCFullYear());
    if (type === 1) return date.toLocaleDateString("en-GB", {month: "short", year: "2-digit", timeZone: "UTC"});
    if (type === 2) return date.toLocaleDateString("en-GB", {day: "2-digit", month: "short", timeZone: "UTC"});
    return date.toISOString().slice(11, type === 4 ? 19 : 16);
  }
  function create(target, elements, options = {}) {
    const lib = window.LightweightCharts;
    if (!lib) throw new Error("The local chart library could not load. Refresh the page.");
    const chart = lib.createChart(target, {
      autoSize: true,
      layout: {background: {type: "solid", color: "#fbfdff"}, textColor: "#496680", fontSize: 11,
        fontFamily: "system-ui, sans-serif", attributionLogo: true,
        panes: {separatorColor: "#e4ecf4", separatorHoverColor: "#c9d9ee", enableResize: true}},
      grid: {vertLines: {color: "#edf2f7"}, horzLines: {color: "#e7eef5"}},
      rightPriceScale: {borderColor: "#d7e2ed", minimumWidth: 82, scaleMargins: {top: 0.12, bottom: 0.08}},
      leftPriceScale: {visible: false},
      timeScale: {borderColor: "#d7e2ed", timeVisible: true, secondsVisible: false,
        rightOffset: 3, minBarSpacing: options.compare ? 0.001 : 0.05, tickMarkFormatter: tickTime},
      localization: {locale: "en-US", timeFormatter: formatTime},
      crosshair: {mode: lib.CrosshairMode.Normal,
        vertLine: {color: "#7b8fa7", labelBackgroundColor: "#39526c"},
        horzLine: {color: "#7b8fa7", labelBackgroundColor: "#39526c"}},
      handleScroll: {mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false},
      handleScale: {mouseWheel: true, pinch: true, axisPressedMouseMove: true, axisDoubleClickReset: true},
    });
    const prices = chart.addSeries(lib.CandlestickSeries, {
      upColor: colors.up, downColor: colors.down, borderVisible: false,
      wickUpColor: colors.up, wickDownColor: colors.down, priceLineVisible: false});
    const volumes = chart.addSeries(lib.HistogramSeries, {
      priceFormat: {type: "volume"}, priceLineVisible: false, lastValueVisible: false,
      autoscaleInfoProvider: (original) => {
        const info = original();
        if (info) info.priceRange.minValue = 0;
        return info;
      },
    }, 1);
    volumes.priceScale().applyOptions({scaleMargins: {top: 0.25, bottom: 0}});
    const forecastPrices = options.compare ? chart.addSeries(lib.CandlestickSeries, {
      upColor: "#6775c830", downColor: "#6775c830", borderVisible: true,
      borderUpColor: colors.forecast, borderDownColor: colors.forecast,
      wickUpColor: colors.forecast, wickDownColor: colors.forecast,
      priceLineVisible: false, lastValueVisible: false}) : null;
    const forecastVolumes = options.compare ? chart.addSeries(lib.HistogramSeries, {
      priceFormat: {type: "volume"}, color: "#6775c870", priceLineVisible: false, lastValueVisible: false}, 1) : null;
    const markers = options.compare ? lib.createSeriesMarkers(prices, []) : null;
    chart.panes()[0].setStretchFactor(3);
    chart.panes()[1].setStretchFactor(1);
    let data = null, rows = [], byTime = new Map(), forecastByTime = new Map();
    function readout(row) {
      if (!row) { elements.readout.textContent = "Move the crosshair over a candle to inspect OHLC and volume."; return; }
      const digits = prices.options().priceFormat.precision;
      const price = (value) => Number(value).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits});
      const prediction = forecastByTime.get(row.time);
      const labels = [
        (row.forecast ? "Forecast" : "History") + " · " + formatTime(row.time),
        "O " + price(row.open), "H " + price(row.high), "L " + price(row.low), "C " + price(row.close),
        "Vol " + Number(row.volume).toLocaleString("en-US", {maximumFractionDigits: 4}) + " " + data.coin,
      ];
      if (prediction && !row.forecast) labels.push("Forecast C " + price(prediction.close),
        "Forecast vol " + Number(prediction.volume).toLocaleString("en-US", {maximumFractionDigits: 4}) + " " + data.coin);
      elements.readout.replaceChildren(...labels.map((text) => { const span = document.createElement("span"); span.textContent = text; return span; }));
    }
    function shadeWindow(region, start, end, extendToEdge = false) {
      if (!region) return;
      if (!start || !end) { region.hidden = true; return; }
      const scale = chart.timeScale(), halfBar = scale.options().barSpacing / 2;
      const first = scale.timeToCoordinate(Date.parse(start) / 1000);
      const last = scale.timeToCoordinate(Date.parse(end) / 1000);
      if (first === null || last === null) { region.hidden = true; return; }
      const plotRight = target.clientWidth - chart.priceScale("right").width();
      const boundary = first - halfBar, endBoundary = extendToEdge ? plotRight : last + halfBar;
      const left = Math.max(0, Math.min(plotRight, boundary));
      const right = Math.max(0, Math.min(plotRight, endBoundary));
      region.hidden = right <= left;
      region.style.left = left + "px";
      region.style.width = Math.max(0, right - left) + "px";
      region.style.bottom = chart.timeScale().height() + "px";
      region.classList.toggle("boundary-offscreen", boundary < 0);
      region.classList.toggle("end-offscreen", endBoundary > plotRight);
    }
    function overlays() {
      elements.volumeTitle.textContent = "Volume (" + (data?.coin || "base asset") + ")";
      elements.volumeTitle.style.top = chart.panes()[0].getHeight() + 10 + "px";
      shadeWindow(elements.forecastRegion, data?.forecast?.[0]?.timestamp,
        data?.forecast?.at(-1)?.timestamp, !options.compare);
      shadeWindow(elements.lookbackRegion, data?.lookback_window?.start, data?.lookback_window?.end);
    }
    chart.subscribeCrosshairMove((event) => readout(byTime.get(event.time) || forecastByTime.get(event.time) || rows.at(-1)));
    if (options.onSelect) {
      let pointerStart = null, dragged = false;
      target.addEventListener("pointerdown", (event) => { pointerStart = [event.clientX, event.clientY]; dragged = false; }, true);
      target.addEventListener("pointermove", (event) => {
        if (pointerStart && Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) dragged = true;
      }, true);
      target.addEventListener("pointerup", () => { pointerStart = null; }, true);
      chart.subscribeClick((event) => {
        const candle = byTime.get(event.time);
        if (!dragged && event.paneIndex === 0 && candle && !candle.forecast) options.onSelect(candle);
      });
    }
    chart.timeScale().subscribeVisibleLogicalRangeChange(overlays);
    chart.timeScale().subscribeSizeChange(overlays);
    // Reposition the volume heading when the pane separator is dragged.
    new ResizeObserver(overlays).observe(chart.panes()[0].getHTMLElement());
    function fit() {
      chart.priceScale("right", 0).applyOptions({autoScale: true});
      volumes.priceScale().applyOptions({autoScale: true});
      chart.timeScale().fitContent(); overlays();
    }
    function focusForecast() {
      if (!data?.forecast?.length) return fit();
      chart.priceScale("right", 0).applyOptions({autoScale: true});
      const first = options.compare ? rows.findIndex((row) => row.time >= Date.parse(data.forecast[0].timestamp) / 1000) : data.history.length;
      const last = options.compare ? rows.findIndex((row) => row.time >= Date.parse(data.forecast.at(-1).timestamp) / 1000) : rows.length;
      const inputStart = options.compare && data.lookback_window ? rows.findIndex((row) => row.time >= Date.parse(data.lookback_window.start) / 1000) : -1;
      chart.timeScale().setVisibleLogicalRange({from: inputStart >= 0 ? Math.max(0, inputStart - 2) : Math.max(0, (first < 0 ? rows.length : first) - 24),
        to: (last < 0 ? rows.length + data.forecast.length : last) + 2});
    }
    function zoom(factor) {
      const range = chart.timeScale().getVisibleLogicalRange();
      if (!range || !rows.length) return;
      const middle = (range.from + range.to) / 2;
      const half = Math.max(4, Math.min(rows.length * 1.5, (range.to - range.from) * factor / 2));
      chart.timeScale().setVisibleLogicalRange({from: middle - half, to: middle + half});
    }
    elements.fit.addEventListener("click", fit);
    elements.focus.addEventListener("click", focusForecast);
    elements.zoomIn.addEventListener("click", () => zoom(0.7));
    elements.zoomOut.addEventListener("click", () => zoom(1.4));
    target.addEventListener("keydown", (event) => {
      if (["+", "=", "-", "ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        if (["+", "="].includes(event.key)) zoom(0.7);
        else if (event.key === "-") zoom(1.4);
        else if (event.key === "Home") fit();
        else if (event.key === "End") focusForecast();
        else {
          const range = chart.timeScale().getVisibleLogicalRange();
          if (range) {
            const shift = (range.to - range.from) * (event.key === "ArrowLeft" ? -0.15 : 0.15);
            chart.timeScale().setVisibleLogicalRange({from: range.from + shift, to: range.to + shift});
          }
        }
      }
    });
    return {setData(value, {preserveRange = false} = {}) {
      const range = preserveRange ? chart.timeScale().getVisibleLogicalRange() : null;
      data = value;
      if (elements.lookbackRegion) elements.lookbackRegion.firstElementChild.textContent = value.lookback_window
        ? `Lookback · ${value.lookback_window.count} candles` : "Lookback window";
      const series = seriesData(options.compare ? {...value, forecast: []} : value);
      rows = series.rows; byTime = new Map(rows.map((row) => [row.time, row]));
      const predicted = seriesData({history: [], forecast: value.forecast || []});
      forecastByTime = options.compare ? new Map(predicted.rows.map((row) => [row.time, row])) : new Map();
      const format = priceFormat([...rows, ...predicted.rows]);
      prices.applyOptions({priceFormat: format});
      prices.setData(series.prices); volumes.setData(series.volumes);
      if (options.compare) {
        forecastPrices.applyOptions({priceFormat: format});
        forecastPrices.setData(predicted.prices.map((candle) => ({...candle, color: "#6775c830"})));
        forecastVolumes.setData(predicted.volumes.map((candle) => ({...candle, color: "#6775c870"})));
        markers.setMarkers(value.selected_candle ? [{time: Date.parse(value.selected_candle) / 1000,
          position: "aboveBar", color: "#39526c", shape: "arrowDown", text: "Last input"}] : []);
      }
      target.setAttribute("aria-label", `${value.coin} price in USDT and volume in ${value.coin}, with UTC candle opening times. ${value.history.length} historical and ${(value.forecast || []).length} forecast candles.`);
      elements.focus.disabled = !(value.forecast || []).length;
      elements.fit.disabled = elements.zoomIn.disabled = elements.zoomOut.disabled = !rows.length;
      fit();
      if (range) chart.timeScale().setVisibleLogicalRange(range);
      else if (!options.compare && rows.length > 180) chart.timeScale().setVisibleLogicalRange({
        from: Math.max(0, value.forecast?.length ? value.history.length - 60 : rows.length - 180),
        to: rows.length + 2,
      });
      readout(rows.at(-1)); requestAnimationFrame(overlays);
    }};
  }
  const api = {seriesData, priceFormat, formatTime, tickTime, create};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else window.KronosChart = api;
})();
