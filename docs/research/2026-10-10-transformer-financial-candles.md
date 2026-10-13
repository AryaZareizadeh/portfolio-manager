# Transformer models for financial candle generation

Research date: October 10, 2026. This review connects primary research papers, official repositories and model cards to the current cryptocurrency forecasting project. Recommendations below are proposed experiments, not measured results on this project's data.

**Recommendation: evaluate Kronos first.** It closely implements the proposed mechanism: historical price and trading activity become discrete tokens, a causal Transformer generates future tokens, and a decoder reconstructs future candles. Start with short four-hour forecasts, evaluate the released model, then fine-tune it and compare with a small model trained locally.

## What the proposed model would learn

Let each candle be an ordered observation containing open, high, low, close and volume. The objective is a conditional distribution over the next candle given completed historical candles. At inference, generate one candle, append it to history, and repeat. During training, actual previous candles can supply the context; a causal attention mask prevents access to later observations.

There are three relevant implementations. **Discrete autoregression** learns a vocabulary of numerical bins or candle codes and predicts token probabilities. **Continuous autoregression** embeds numerical candles or patches and predicts a numerical distribution or values. **Direct forecasting** predicts a future horizon together. All can use Transformers; only the first requires a discrete vocabulary. A candle can have several tokens, and a token can represent several time steps.

Use numerical OHLCV records as the initial representation. The existing CSV data provides exact values, units and timestamps without needing chart-image extraction. Joint candle prediction matters: forecasting close and volume independently does not capture their dependence or ensure that high and low contain open and close.

## Closest financial models

| Model | Forecasting mechanism | Fit to this project |
| --- | --- | --- |
| **Kronos** | Learned hierarchical candle tokens; decoder-only autoregressive prediction | Closest match to joint price/activity candle generation; first candidate |
| **StockGPT** | Binned stock returns; next-token decoder | Clear small-model precedent, but no full OHLCV output |
| **FinCast** | Continuous numerical patches; causal Transformer with sparse experts | Financial forecasting comparator; channels forecast independently |
| **KiT** | Diffusion Transformer with flow matching over future OHLCV paths | Joint path-generation alternative; emerging release with documentation conflicts |

**Kronos (2025; AAAI 2026).** Its learned tokenizer uses Binary Spherical Quantization to encode OHLCV plus turnover amount into coarse and fine subtokens. A decoder predicts these subtokens with cross-entropy and reconstructs future observations. The paper reports over 12 billion training candles across 45 exchanges and seven resolutions. Its training period ends June 2024 and evaluation begins July 2024. Binance cryptocurrency is included in evaluation, but cryptocurrency and forex experiments omit volume and amount. Thus the paper does not directly establish the added value of crypto-volume conditioning. Its four-hour evaluation horizon is 18 bars; a 180-bar forecast is a substantial extrapolation. Price-series correlations and reported trading simulations address different tasks from accurately predicting a full crypto path. [Kronos paper](https://arxiv.org/html/2508.02739v1)

Released variants are mini, 4.1M parameters with 2,048 context positions; small, 24.7M with 512; and base, 102.3M with 512. The 499.2M large variant is unreleased. The repository is MIT-licensed and supplies matching tokenizers, inference and fine-tuning workflows. The predictor accepts OHLC, optional volume/amount and historical/future timestamps. Multiple generated paths are averaged by its public `sample_count` interface. Retaining individual scenarios requires separate single-sample calls or adapting the lower-level interface. [Official repository](https://github.com/shiyu-coder/Kronos), [CSV fine-tuning workflow](https://github.com/shiyu-coder/Kronos/blob/master/finetune_csv/README.md)

There is a practical horizon limit in the reviewed predictor code: generation uses sliding token buffers, but reconstruction decodes only the final `max_context` positions. A longer request with a full future timestamp index then fails DataFrame construction because too few rows were decoded. Missing amount with available volume is approximated as volume times mean OHLC; this is not observed turnover. Volume absence fills both activity fields with zero. Preserve that distinction in dataset provenance. [Reviewed predictor source](https://github.com/shiyu-coder/Kronos/blob/master/model/kronos.py#L350-L501)

**StockGPT (2024).** This decoder-only model predicts distributions over 402 daily-return bins using 256 historical observations. Its approximately 0.93M parameters show that a financial autoregressive experiment can be small. Training and testing use long histories of US equities; the data scale and equity portfolio results do not establish crypto candle performance. It predicts returns rather than open/high/low/volume, so it is a design reference for a controlled local experiment. [StockGPT paper](https://arxiv.org/html/2404.05101v3)

**FinCast (2025).** A financial 1B-parameter model forecasts continuous numerical patches with a sparse causal Transformer, frequency embeddings and point/quantile loss. Its shared univariate forecaster is applied independently to channels, so it does not provide joint OHLCV modeling by itself. Pretraining covers financial and nonfinancial sources, including cryptocurrencies. Authors report consumer-GPU inference, but this review did not benchmark memory or runtime locally. The released PEFT code remains described as needing testing. It is a useful comparator, with Apache-2.0 licensing, rather than the closest implementation of next-candle token classification. [Paper](https://arxiv.org/html/2508.19609v1), [Official repository](https://github.com/vincent05r/FinCast-fts), [Weights](https://huggingface.co/Vincent05R/FinCast)

**KiT (September 28, 2026).** This recent model generates conditional OHLCV trajectories jointly through flow matching rather than one-token-at-a-time decoding. Its gap/body/shadow/volume representation preserves candle geometry; the paper covers multiple equity markets and crypto. However, the paper specifies a December 2023 training cutoff while the released three-market checkpoint card specifies December 2025. This discrepancy needs resolution before interpreting its evaluation as reproducible. Repository training instructions remain incomplete, and code is AGPL-3.0. Treat it as a later path-generation comparison. [Paper](https://arxiv.org/html/2609.34507v1), [Official repository](https://github.com/Luciferbobo/KiT), [Checkpoint documentation](https://huggingface.co/Lucifer744/KiT_model)

## General time-series Transformers

| Model/version | Mechanism and output | Relevance and availability |
| --- | --- | --- |
| **Original Chronos-T5** | Quantizes scaled values into 4,096 tokens; autoregressively samples tokens and decodes them into values | Strong demonstration of numerical token prediction; native univariate forecasting. Public Apache-2.0 weights. [Model card](https://huggingface.co/amazon/chronos-t5-base) |
| **Chronos-Bolt** | Continuous patches and direct multi-step quantiles | Univariate; does not use the original Chronos token-sampling mechanism. Apache-2.0. [Model card](https://huggingface.co/amazon/chronos-bolt-base) |
| **Chronos-2** | 120M encoder-only model with attention across grouped series; direct quantile forecasts | Native multivariate targets/covariates; 8,192 context and up to 1,024 forecast steps. Public Apache-2.0 weights; quantiles do not supply a coherent sampled joint candle path by themselves. [Model card](https://huggingface.co/amazon/chronos-2), [Paper](https://arxiv.org/abs/2510.15821) |
| **TimesFM 2.5** | Decoder-only Transformer with numerical patch decoding and point/quantile forecasts | Native univariate model with external covariate support; Apache-2.0 weights/code. [Model card](https://huggingface.co/google/timesfm-2.5-200m-pytorch) |
| **TimesFM 3.0, August 2026** | 330M model with temporal/variable attention; fills a masked future horizon in one pass | Native multivariate support; non-autoregressive inference. Downloaded/self-hosted weights prohibit commercial and production use; authorized Google Cloud services permit these uses. Versions through 2.5 use Apache-2.0. [Architecture](https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [License notice](https://github.com/google-research/timesfm#license-notice-for-pretrained-weights) |
| **Lag-Llama** | Decoder-only model predicts Student's t distribution parameters and generates numerical trajectories autoregressively | Public Apache-2.0 weights; native univariate model. A probabilistic return baseline rather than a ready joint OHLCV generator. [Paper](https://arxiv.org/html/2310.08278v3), [Official repository](https://github.com/time-series-foundation-models/lag-llama) |

Multivariate support helps model dependencies, but does not automatically enforce candle geometry or supply calibrated joint scenarios. This is an inference from the output formulations, not a claim that these models were benchmarked on this project.

## What the evidence establishes

Results depend on the task and domain. **Re(Visiting) Time Series Foundation Models in Finance** compares generic pretraining, fine-tuning and finance-native pretraining on daily excess returns. Its authors find that off-the-shelf models and their fine-tuned variants underperform strong benchmarks, while financial pretraining improves results. Those findings motivate domain-specific training, but daily equity excess returns differ from intraday crypto OHLCV. [Study](https://arxiv.org/abs/2511.18578)

The accompanying FinText release includes Chronos and TimesFM models tied to historical training years. This highlights a valuable evaluation practice: use a checkpoint whose training information was available at the simulated forecast date. The released workflows operate on return series, not joint candles. [FinText model collection](https://huggingface.co/FinText), [Official implementation](https://github.com/DeepIntoStreams/TSFM_Finance)

A separate 2026 benchmark on five US equities finds pretrained models frequently rank above locally trained neural models, while statistically supported improvements over random-walk baselines remain limited. The authors explicitly do not establish trading profitability. Its narrow asset/task coverage limits transfer to this project. [Benchmark](https://arxiv.org/html/2606.27100v1)

Transformer architecture alone is not evidence of a forecasting advantage. Earlier LTSF-Linear research found simple linear models competitive with or superior to the Transformer variants tested on general forecasting datasets. This is an argument for retaining strong simple baselines, not a verdict against later financial Transformers. [LTSF-Linear study](https://arxiv.org/abs/2205.13504)

## Connection to the existing project

The current [model implementation](D:/GitHub/portfolio-manager/crypto_forecast/model.py:107) uses separate GRU/LSTM encoders for 15m, 1h, 4h and daily streams. Each reads [11 continuous causal features](D:/GitHub/portfolio-manager/crypto_forecast/multiframe.py:15). Its output is three cumulative log returns at 1, 7 and 30 days, not future OHLCV observations. [Target construction](D:/GitHub/portfolio-manager/crypto_forecast/multiframe.py:145) and [prediction](D:/GitHub/portfolio-manager/crypto_forecast/pipeline.py:120) confirm there is no candle rollout.

Useful existing infrastructure includes completed-candle alignment, OHLCV validation, gap exclusion, source provenance, chronological splitting, purged labels and training-only normalization. Replacing a recurrent encoder with attention would still leave the current three-target task. Candle generation changes the representation, labels, output distribution, loss, sampling loop and evaluation.

The [recorded five-coin experiment](D:/GitHub/portfolio-manager/docs/experiments/2026-10-10-multiframe-coins.md:7) used 202,345 native 15m rows per coin and about 32,615 aligned training samples. Its reported GRU MAE was worse than unchanged price at all three horizons. These are recorded results, not experiments rerun for this review. A million overlapping bars across correlated coins should not be treated as a million independent examples. Existing inspected test data can be reused for development comparisons, but new final claims require an untouched evaluation period.

## Proposed first experiment

1. **Use one native prediction frequency initially: four-hour candles.** Keep the ingestion and causal alignment machinery. Multiple conditioning resolutions can be an ablation after a single-stream experiment is established.
2. **Evaluate a released Kronos model, then fine-tune Kronos-small.** Mini is a smaller feasibility option; choose the matching tokenizer. Keep volume and its units explicit, and identify approximated amount. Compare pretrained and adapted versions on identical forecast origins.
3. **Add a small local autoregressive model as a controlled comparison.** Use either learned discrete codes with cross-entropy or continuous candle embeddings with a probabilistic output. For locally fitted codes, learn the tokenizer only from training data. A continuous model can test the sequence idea without first building a codebook.
4. **Evaluate short horizons before long rollouts.** At four-hour resolution, 1/7/30 days mean 6/42/180 steps. At fifteen-minute resolution they mean 96/672/2,880. The 180-step case fits Kronos-small's reviewed 512-position reconstruction limit, but is ten times its paper's four-hour evaluation horizon. Compatibility is not accuracy evidence.
5. **Retain scenarios and compare distributions.** Generate separate paths, summarize endpoint quantiles and evaluate their calibration. Averaging paths can obscure scenario structure; interpolating endpoint quantiles does not create a jointly plausible path.

For a custom model, a useful proposed candle representation is the log opening gap relative to previous close, log close/open body, nonnegative upper/lower log shadows, and log(1 + volume). Reconstructing with positive prices and nonnegative shadows/volume can enforce `low <= min(open, close) <= max(open, close) <= high`. This preserves real opening gaps rather than forcing every open to equal the previous close. The representation is an implementation proposal to test, not a validated improvement.

Evaluate next-bar returns, ranges and volume, then multi-step path error, interval coverage, proper probabilistic scores and invalid-candle frequency. Compare return point forecasts with unchanged price, trailing drift and the existing GRU/LSTM; add a suitable return/volatility scenario baseline for distribution metrics. Keep true future values out of autoregressive rollout evaluation. For probabilistic scores, do not report a quantized token likelihood as if it were a continuous density without accounting for the quantization scheme.

Use common chronological boundaries across coins; fit local preprocessing on training data; purge labels spanning split boundaries; add rolling evaluations across market regimes and an untouched final period. Audit the pretrained checkpoint's actual data cutoff as well as local fine-tuning dates. Any later strategy evaluation must include fees, spread, slippage and turnover. A plausible generated chart and a better forecasting score establish different things.

This review created documentation only. It did not install model packages, download checkpoints, alter application code, run training or establish local hardware performance.
