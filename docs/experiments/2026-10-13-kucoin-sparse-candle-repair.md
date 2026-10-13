# KuCoin sparse candle aggregation repair — 2026-10-13

LTC's first saved 4h candle was 2018-01-30 16:00 UTC, although native 15m candles began 2018-01-29 00:00 UTC. Requiring every 15m slot incorrectly discarded real larger candles because KuCoin omits intervals with no ticks. Successful bounded pages cover those absent slots; failed pages and uncovered periods do not.

[KuCoin's official Klines documentation](https://www.kucoin.com/docs-new/rest/spot-trading/market-data/get-klines) describes no-tick omissions and its 1,500-row limit. The fix aggregates observed OHLCV only within fully checked, bounded response ranges. It fills no prices or source rows. Unknown coverage and other exchanges retain strict slot completeness. Entire buckets with no observed trades remain absent. Listing-edge buckets require coverage for their entire interval. Incremental downloads reuse verified no-tick coverage; recent refresh accepts sparse successful KuCoin pages.

All first six repaired LTC 4h candles on 2018-01-29 were independently compared with KuCoin's native `4hour` endpoint. Their OHLCV match, including the first candle: open 194.100022, high 198.9, low 184.000001, close 187.25, volume 98.640824. The native evidence URL and response hash are preserved in `.test-tmp/candle-gap-repair-20261013/ltc-native-4h-evidence.json`.

Only LTC and DASH changed among 21 local KuCoin histories. Each now starts at its first fully covered bucket: LTC 2018-01-29 00:00 UTC; DASH 2018-09-06 00:00 UTC. All native 15m bytes and hashes stayed unchanged, as did every existing derived CSV row's exact text. Local raw response hashes were checked for 204 LTC and 484 DASH pages. One uncached DASH recent response was excluded from offline sparse-coverage authorization.

Missing counts below use the same post-listing time window before and after repair; they exclude pre-listing dates.

| Coin | Interval | Rows before | Rows after | Missing before | Missing after |
| --- | --- | ---: | ---: | ---: | ---: |
| DASH | 1d | 2,947 | 2,959 | 12 | 0 |
| DASH | 1h | 70,756 | 70,849 | 265 | 172 |
| DASH | 4h | 17,688 | 17,739 | 67 | 16 |
| LTC | 1d | 2,796 | 3,179 | 383 | 0 |
| LTC | 1h | 69,605 | 75,309 | 6,696 | 992 |
| LTC | 4h | 16,969 | 19,067 | 2,106 | 8 |

The remaining empty higher buckets have no observed trades and are kept absent. The 15m availability reports still show omitted source slots and pre-listing dates. The selected exchange is unchanged; the earlier Binance monthly archive grid-validation failure recorded in LTC's source comparison remains visible.

Exact prior derived CSV/metadata and download report bytes are backed up in `.test-tmp/candle-gap-repair-20261013/`. `audit.json` records hashes, comparable coverage counts, and verification. `data/download_report.json` now matches updated sidecars. Loaders verified all repaired files and source hash links after publication. No training job was active when files were published.

Wholly empty pages remain retryable and do not independently authorize sparse aggregation. This preserves recovery when an empty response is transient; nonempty historical pages still cover their omitted no-tick slots.

Regression coverage includes sparse OHLCV without filling, incomplete/adjacent page boundaries, source/instrument/hash manifest checks, download persistence, no-tick recent refresh, failed/schema/duplicate response preservation, and avoiding repeated no-tick downloads. The focused data suite passed 89 tests.

The combined worker/data full regression suite passed **542 tests in 111.91 seconds**. Independent review also verified source hashes, preservation of existing derived rows, loader provenance links, and the native LTC comparison.
