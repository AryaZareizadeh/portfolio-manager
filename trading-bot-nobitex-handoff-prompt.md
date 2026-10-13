# Project handoff: adaptive spot portfolio management

I want to extend my existing `Trading-bot-Nobitex` project with an adaptive, machine-learning-based spot portfolio manager. This prompt records the requirements and research from a separate design conversation. Treat the confirmed requirements as settled; ask only about consequential gaps that cannot be resolved by inspecting the project or making a clearly stated, reversible assumption.

The repository is `D:\GitHub\Trading-bot-Nobitex`.

## 1. Objective and confirmed requirements

Build a system that grows the total value of a managed portfolio measured in **Tether USDT, after trading costs**. The initial research, paper-trading, and eventual live portfolio size is approximately **200 USDT**.

- I select the allowed assets. The system decides which subset to hold and how much to allocate to each.
- Assets may receive zero allocation. Holdings do not need equal weights, and the system does not need to own every approved asset.
- USDT is the accounting unit and reserve asset. The system may hold up to **100% USDT** when opportunities are unattractive.
- Holding periods must adapt to market conditions. Evaluate several forecasting horizons and reassess positions over time; do not assume one fixed holding period for all trades.
- The portfolio drawdown target is **10%**. This replaces an earlier 5% preference. It applies to the managed portfolio's peak-to-trough USDT equity, including unrealized gains/losses and costs, not to each position separately.
- This is a risk target, not a guaranteed cap. Gaps, liquidity shortages, and execution failures can cause larger losses.
- The first version is **spot only**, with no borrowing, short selling, leverage, margin renewal fees, or liquidation accounting.
- Eventually, the system should execute automatically on **Nobitex** without asking for approval of every trade. However, the immediate scope is research, backtesting, and paper trading. Do not place real orders, enable live trading, or change existing live bots as part of this work. I will explicitly enable real-money operation later after reviewing validation results.
- Keep the design extensible so other trading methods can be added later with their own accounting, execution, and risk rules.
- Replace the existing **Portfolio** tab with a **Portfolio Manager** tab. This is the dashboard home for the new portfolio manager, replacing the existing navigation entry rather than adding a second portfolio tab.

Approved universe:

`BTC, ETH, USDT, BNB, XRP, SOL, ZEC, HYPE, ADA, XLM, NEAR, UNI, LTC, SUI, ENA, XAUT`

This is 15 candidate assets plus USDT. Keep the list configurable. XAUT is gold exposure and must be evaluated as a risky asset, not treated as equivalent to USDT. There is no approved fixed return target, mandatory minimum number of holdings, or requirement to trade continuously.

## 2. Inspect and integrate with the existing project

A read-only inspection on September 27, 2026 found:

- Python 3.13, FastAPI, a plain JavaScript dashboard, SQLite persistence, Docker deployment, and per-user account isolation.
- `price_ai.py` already contains a simulation-only `PriceAI` using scikit-learn `HistGradientBoostingRegressor`. It predicts four-candle returns, uses a fixed training boundary, and supports long/short decisions through the existing simulator. This is a useful baseline, but it is not yet the requested multi-asset spot allocator.
- `data.py` retrieves Nobitex/Binance candles.
- `execution_model.py`, `accounting.py`, `backtester.py`, and `optimizer.py` provide research and accounting infrastructure, much of it oriented around margin positions.
- `portfolio.py` and the Portfolio tab already provide USDT valuation, cash-flow-adjusted performance, manually supplied targets, and indicative rebalancing ideas. The inspected target validation caps each risky asset at 35% and requires at least 20% USDT; these are existing implementation policies, not preferences established in this conversation.
- `exchange_execution.py` provides durable order-intent/reconciliation patterns, but its existing execution path is margin-oriented.
- The repository had substantial uncommitted work. Preserve and account for it. Do not overwrite changes, reset the checkout, or assume the working tree is clean.
- Some documentation disagrees with newer code. Verify current behavior in code and tests before deciding what to reuse. For Nobitex API work, check the repository's local API index and matched references as well as current official documentation.

Extend the existing application rather than building an unrelated replacement. Reuse suitable data, persistence, UI, authentication, and research components. Add explicit spot accounting where the current margin model is unsuitable. A leverage value of 1 does not, by itself, turn a margin simulator into a spot simulator.

Evolve the existing Portfolio tab into **Portfolio Manager**. Bring the approved-asset settings, managed budget, training and validation results, allocation decisions, paper-trading controls, and performance reporting together in this tab. Reuse relevant valuation and cash-flow history, preserving saved data during the transition. Update navigation labels and affected UI tests to reflect the replacement.

Keep the new manager's 200 USDT budget, holdings, and performance separate from unrelated wallet balances and existing bots. Preserve per-user isolation. Future live execution must prevent competing bots from spending the same assets.

## 3. Data sources and exchange separation

Use **Binance as the preferred primary market-signal source** because I want broader market price/volume information. **Nobitex remains the execution venue** and the source of actual tradable prices, order-book liquidity, spreads, fees, balances, and fills.

Nobitex's lower volume does not make its executable prices irrelevant. A Binance forecast must be evaluated against what can actually be bought and sold on Nobitex. Track cross-exchange price differences without assuming they always converge.

I have authorized additional reliable historical spot-data sources when Binance history is insufficient:

| Asset | Historical source | Evidence verified in the design conversation |
|---|---|---|
| HYPE/USDT | KuCoin spot | Public API returned hourly candles beginning December 7, 2024 at 15:00 UTC, and daily candles from the launch period. |
| XAUT/USDT | Bitfinex spot | Public API returned hourly candles beginning January 25, 2020 at 09:00 UTC, daily candles from January 2020, and recent September 2026 candles. Bitfinex identifies this pair as `tXAUT:UST`; its `UST` symbol means Tether USDT. |

These were successful sample retrievals, not a full audit of continuous history. Early Bitfinex XAUT hourly data was sparse. Binance's official announcements placed its HYPE spot listing on September 24, 2026 and XAUT spot listing on March 26, 2026, which motivated the supplementary sources.

Official references:

- Binance public market data: https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md
- KuCoin current historical candle API: https://www.kucoin.com/docs-new/v2/rest/ua/get-klines
- KuCoin spot listing: https://www.kucoin.com/announcement/en-hyperliquid-hype-gets-listed-on-kucoin
- Bitfinex candles: https://docs.bitfinex.com/reference/rest-public-candles
- Bitfinex pair metadata: https://api-pub.bitfinex.com/v2/conf/pub:list:pair:exchange
- Nobitex API: https://apidocs.nobitex.ir/
- Nobitex fees: https://nobitex.ir/pricing/

Sample requests that succeeded during the research:

```text
https://api.kucoin.com/api/v1/market/candles?symbol=HYPE-USDT&type=1hour&startAt=1733443200&endAt=1733702400
https://api-pub.bitfinex.com/v2/candles/trade:1h:tXAUT:UST/hist?sort=1&start=1546300800000&limit=5
```

Recheck API versions, schemas, limits, market availability, and supported intervals before implementation. KuCoin's current documentation describes spot history with time pagination and up to 1,500 records per request; Bitfinex documents up to 10,000 candles per request. The tested KuCoin request used its classic endpoint, so do not assume its schema matches the newer API.

Data requirements:

- Record exchange, instrument, quote currency, timestamps, retrieval time, and data version/hash. Normalize timestamps to UTC and explicitly map candle fields; providers use different field orders and time units.
- Audit gaps, duplicates, outliers, incomplete candles, sparse trading, and actual historical availability. Do not fabricate pre-listing history or treat unknown intervals as evidence of tradability.
- Use spot history for the initial model. Do not silently substitute perpetual-futures prices, generic gold prices, or PAXG for XAUT.
- Keep source provenance and normalize volume within each exchange. Do not blindly splice raw volumes or create false returns at a source switch.
- Compare overlapping periods across sources and evaluate whether forecasts trained on KuCoin/Bitfinex transfer to Binance live inputs. Preserve supplementary live feeds during paper validation when useful.
- Allow each asset to become eligible when sufficient valid history exists. Do not shorten every asset's training history to match the newest listing.
- Verify Nobitex market availability and minimum order/quantity rules. Prefer direct USDT pairs for the initial design; flag assets that would require an additional conversion route rather than silently introducing IRT exposure.
- Record Nobitex order-book data for forward simulation. If historical Nobitex depth is unavailable, label historical execution results as estimates and stress-test their assumptions.

## 4. Recommended modeling approach

The recommended first research candidate is **gradient-boosted tree forecasts across multiple horizons, paired with a constrained portfolio allocator and independent risk controls**. LightGBM is the preferred candidate, not an already proven winner. Compare it fairly with the existing HistGradientBoosting model and simple baselines. Do not discard working infrastructure merely to change libraries.

Separate these responsibilities:

1. Market data and quality checks.
2. Features and return/downside forecasts.
3. Portfolio allocation and turnover decisions.
4. Independent risk checks.
5. Historical, paper, and eventual live execution adapters.
6. Portfolio accounting, model versioning, and reporting.

Use a manageable feature set: returns, momentum, relative strength, volatility, volume changes, liquidity where available, and broader market context. Evaluate pooled cross-asset learning while allowing for asset differences, especially XAUT.

Use multiple lookback timeframes and multiple forecast horizons. Example candidates are 15-minute, hourly, four-hour, and daily inputs, with forecasts over several hours through several days. These are starting hypotheses for validation, not user-approved constants. Choose a tractable initial set using training/validation data and execution costs.

Forecast return magnitude and downside/uncertainty as well as relative attractiveness. A ranking alone cannot establish that switching pays for itself or that buying beats holding USDT. Assess uncertainty estimates on held-out data; do not label uncalibrated scores as probabilities.

Make holding periods adaptive through repeated decisions using currently available information. Do not select each trade's best horizon retrospectively. Distinguish live monitoring frequency, allocation decision frequency, and retraining frequency.

Do not make reinforcement learning, a large neural network, or an LLM the initial dependency. More complex models may be evaluated later if they demonstrate improvement under the same validation and cost assumptions.

## 5. Allocation, spot accounting, and risk

The allocator chooses nonnegative weights across eligible assets and USDT. Total allocations must respect one shared portfolio budget after fees and cash reservations. Maintain actual asset quantities and cash; do not run independent 200 USDT backtests for every coin and combine their profits.

Consider forecast returns, correlated risk, uncertainty, Nobitex liquidity, concentration, and the incremental cost of changing existing holdings. Include a no-trade region so small forecast changes do not cause constant switching. When routing through USDT, account for the costs of both selling the old asset and buying the new one.

Respect minimum order notionals, quantity increments, rounding, and unsellable residual balances. With 200 USDT, meaningful position sizes matter. The approved list is a candidate universe, not a request for 15 simultaneous positions.

USDT earns zero modeled return in USDT unless a separate yield product is explicitly added later. Spot holdings must not incur invented margin/funding fees.

Measure strategy equity and drawdown from its managed holdings using Nobitex-based valuation where available. Distinguish marked equity from estimated liquidation proceeds, make costs explicit, and adjust performance/high-water accounting for external cash flows. At a 200 USDT peak, a 10% drawdown corresponds to 180 USDT; the threshold moves with subsequent peaks.

Propose configurable exposure reduction and halt/re-entry rules around the drawdown target. Do not equate a 10% per-position stop with 10% portfolio protection. Exact concentration limits, cash floors, turnover caps, and re-entry thresholds are design decisions still to justify. Preserve existing behavior for other strategies when making these configurable.

Stale market data, missing valuations, unresolved orders, and unreliable liquidity estimates must prevent new risk-taking. Distinguish pausing new entries from managing existing positions; do not blindly liquidate against an untrusted book. Future live execution must persist intents and reconcile partial fills/timeouts before resubmission.

## 6. Backtesting and simulation are required deliverables

Build the simulator alongside the model. I need evidence I can inspect before granting access to real money.

**Historical research:**

- Use chronological walk-forward training and validation, plus a final untouched test period. Split by time across all assets rather than randomly splitting rows.
- Fit transformations, feature selection, models, thresholds, horizon choices, and portfolio parameters only within the permitted training/validation windows.
- Exclude training labels whose future-return windows overlap an evaluation boundary. Align higher-timeframe features by when their candles were actually complete and available.
- Execute decisions only after the information used to make them existed. Specify fill timing and test for look-ahead leakage.
- Test several market conditions and report per-period/per-asset results. A short profitable interval is insufficient evidence.
- Version experiment settings, data, code/model artifacts, random seeds, costs, and evaluation windows. Track repeated tuning and avoid selecting a model on the final test.
- Include realistic fees, spread, slippage, order minimums, limited liquidity, partial/unfilled orders, delays, and stress scenarios where the available data supports them. State clearly which effects an OHLC-only backtest cannot reconstruct.
- Do not assume a limit order fills merely because a candle touched its price. Treat historical cost schedules as assumptions when only current exchange fees are available.

**Live paper trading:**

- Start with a virtual 200 USDT balance and no real-order capability.
- Use live signal inputs and actual Nobitex public order books to simulate the same forecast/allocation/risk policy.
- Reuse strategy and accounting logic across historical replay and paper trading; isolate the execution adapter.
- Persist virtual cash, holdings, orders, fills, model version, decisions, and reasons across restarts. Follow the existing explicit-start behavior.
- Label simulated fills as estimates, even when based on live depth.

**Comparisons and reports:**

Compare against holding USDT, a precisely defined buy-and-hold approved basket, and a simple momentum/volatility allocation baseline under comparable starting capital and costs. Ensure historical benchmarks respect asset availability. Include the existing PriceAI where comparison is meaningful.

Show net P&L and return in USDT, peak-to-trough drawdown, equity history including open holdings, fees and estimated execution costs, turnover, trade count, holding durations, allocations through time, time in USDT, and reasons for allocation changes or rejected trades. Report forecast quality separately from portfolio profitability. Distinguish simulated results, live results, and operating costs. Do not annualize a short favorable sample as a dependable return expectation.

Define validation acceptance criteria before final evaluation. If no candidate shows credible improvement after costs while reasonably meeting the risk objective, report that outcome. Do not tune until the result appears profitable or claim a guaranteed edge.

## 7. Delivery and working approach

Start by inspecting the current repository, instructions, tests, uncommitted changes, and existing PriceAI/portfolio features. Explain what can be reused, what must change for spot accounting and portfolio-wide decisions, and any material conflicts or missing inputs.

Then produce a concrete architecture and phased implementation plan covering:

1. Source adapters, historical acquisition, and data-quality reporting.
2. Shared spot portfolio accounting and a baseline simulator.
3. Multi-horizon forecasting and allocation experiments.
4. Walk-forward evaluation and reproducible result reports.
5. Live paper trading and replacement of the existing Portfolio tab with the Portfolio Manager tab.
6. A separately gated future Nobitex live-spot adapter.

Keep the first implementation focused on research and paper trading. Do not require exchange credentials to download public training data or run simulations, and do not expose existing secrets while inspecting the project.

Add meaningful tests for temporal leakage, cash conservation, trading costs, source normalization, order constraints, drawdown accounting, multi-asset budget sharing, persistence, and per-user isolation. Preserve existing margin bots and regression coverage. Update documentation to match the implemented behavior.

Present important unresolved design choices with a recommendation. Avoid asking again about requirements settled above, avoid unnecessary infrastructure for a 200 USDT initial portfolio, and distinguish a working implementation from evidence that its strategy is profitable.
