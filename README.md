# Demand Forecasting

A production-style demand-forecasting system on the [M5 (Walmart) dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy): 28-day-ahead daily quantile forecasts for 30,490 item-store series, built as packaged, tested, containerised Python — not a notebook. The forecasting model is deliberately bounded; the engineering around it is the point: honest hierarchical evaluation, batch inference, automated retraining, a shadow → A/B deployment harness, and infrastructure as code.

**Status: Milestone 2 (feature pipeline + EDA) complete.**

## Problem framing

For each product in each store, produce a 28-day-ahead daily demand forecast that downstream replenishment and inventory systems act on. The model does not "predict sales" in the abstract — it sizes an order.

Forecast error is asymmetric in business terms: **under-forecasting** causes stockouts and lost sales; **over-forecasting** causes holding cost, markdowns, and spoilage for perishables. Those costs differ by product and are not the model's job to assume. That is why the system ships a **predictive distribution (quantiles)**, not a single point forecast — the service level (which quantile to order to) is a business choice informed by margin and spoilage.

## Design decisions

| Decision | Rationale |
|---|---|
| **Quantile forecasts, not point forecasts** | Asymmetric cost of under- vs over-forecasting; the business picks the service level. |
| **Global LightGBM baseline** | One gradient-boosted model across all 30,490 series (series identity as features). Strong, fast to retrain, and top-tier in the actual M5 competition — not a strawman. |
| **TFT (Temporal Fusion Transformer) as the upgrade** | Native multi-horizon quantiles + interpretability; the model family used in production for this problem. Compared honestly against the baseline **including retraining cost** — recommending the simpler model if it wins is the intended outcome, not a failure. |
| **WAPE / MASE / WRMSSE, not MAPE** | A large share of item-days have zero sales (intermittent demand); MAPE divides by zero. WAPE weights by volume, MASE compares against a seasonal-naive baseline, WRMSSE is the official M5 metric for public comparability. |
| **Evaluation across hierarchy levels** | Item → department → category, store → state. Aggregate accuracy hides bottom-level failure; orders are placed at item-store level. |
| **Evaluation across horizon (d+1 … d+28)** | Day 1 and day 28 are different decisions with different uncertainty; accuracy is reported as a curve, not an average. |
| **Pinball loss + empirical coverage** | Score the distribution, not just the mean. An interval that is wrong about its own confidence is dangerous for inventory. |
| **Rolling-origin, expanding-window backtesting** | Train to time *t*, forecast 28 days, roll forward, repeat. The time-series-correct analogue of cross-validation and the only honest estimate of production performance. |
| **Point-in-time-correct features, one pipeline** | Every training row uses only information available at that moment (lags, known prices, calendar). The exact same feature code is imported by training and batch inference — no training–serving skew — with a test that fails on future leakage. |
| **Batch inference, no API** | Forecasts for the whole catalogue are produced overnight; throughput matters, latency does not. Every forecast is written to a prediction archive for later scoring against actuals. |
| **Cold-start as a first-class case** | New items have no history, so lag features are undefined; they fall back to hierarchy-level priors (department/category/store) until history accumulates. |
| **Continuous Training + safe rollout** | Retraining runs on a schedule *and* on a monitored-error trigger. New models go shadow → A/B → promote, with rollback — a model change is a deployment, not a file swap. |

## Implementation phases

| Phase | Scope | Status |
|---|---|---|
| 1. Scaffold | Poetry project, CI gate (ruff + mypy + pytest), Dockerfile, data loading + schema validation + Parquet conversion | ✅ done |
| 2. Features + EDA | Point-in-time feature pipeline (lags, rolling stats, calendar, price) + no-leakage test; isolated EDA notebook | ✅ done |
| 3. LightGBM baseline | Config-driven global model (per-quantile), MLflow tracking + registry, rolling-origin backtest | ⏳ next |
| 4. Evaluation panel | WAPE/MASE/WRMSSE × hierarchy level × horizon; pinball loss + calibration report | — |
| 5. TFT upgrade | Darts TFT with native multi-horizon quantiles; honest comparison incl. retraining cost | — |
| 6. Batch inference + monitoring | Scheduled forecast job → prediction archive; data-drift + forecast-error monitoring; cold-start fallback | — |
| 7. Continuous Training | Scheduled + trigger-based automated retraining wired to the monitoring signal | — |
| 8. Deployment harness | Shadow → A/B → promote → rollback on registry stages | — |
| 9. Infrastructure as code | Storage, scheduler, and registry backend defined in Terraform | — |

Phases 1–6 are the MVP: train either model with one command, backtest it honestly, and produce monitored 28-day quantile forecasts for the whole catalogue. Phases 7–9 are the platform layer.

## Quickstart

```bash
poetry install

# Get the data (Kaggle account required; accept the competition rules first):
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/
# unzip so that data/raw/ contains: sales_train_evaluation.csv, calendar.csv, sell_prices.csv

# Validate and convert to Parquet:
poetry run python -m demand_forecasting.data.convert --config config/config.yaml

# Quality gate:
poetry run ruff check src/ tests/ && poetry run mypy src/ && poetry run pytest
```

## Repository structure

```
├── config/config.yaml         # all run parameters in one place
├── src/demand_forecasting/
│   ├── data/                  # loading + schema validation + Parquet conversion
│   ├── features/              # point-in-time feature pipeline (shared)      [phase 2]
│   ├── training/              # LightGBM + TFT + MLflow                      [phases 3, 5]
│   ├── evaluation/            # metric panel + rolling-origin backtest       [phases 3, 4]
│   ├── inference/             # batch forecast job → prediction archive      [phase 6]
│   ├── monitoring/            # drift + forecast-error checks                [phase 6]
│   └── platform/              # retraining trigger + deployment harness      [phases 7, 8]
├── infra/terraform/           # infrastructure as code                       [phase 9]
├── tests/                     # unit + data + no-leakage tests (synthetic fixtures)
└── notebooks/                 # EDA only, clearly separated
```

## Non-goals

Not chasing the M5 leaderboard; not heavy data cleaning; not ensembles or model exotica; not formal hierarchical reconciliation (a simple coherent roll-up is reported instead). The evaluation and operations rigor is the deliverable.

## Stack

Python 3.12 · Poetry · LightGBM · Darts + PyTorch (TFT) · pydantic · MLflow · Evidently · pytest / ruff / mypy · Docker · GitHub Actions · Terraform
