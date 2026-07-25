# Demand Forecasting

A production-style demand-forecasting system on the [M5 (Walmart) dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy): 28-day-ahead daily quantile forecasts for 30,490 item-store series, built as packaged, tested, containerised Python — not a notebook. The forecasting model is deliberately bounded; the engineering around it is the point: honest hierarchical evaluation, batch inference, automated retraining, a shadow → A/B deployment harness, and infrastructure as code.

**The system is built twice.** A **portable core** (`src/`, cloud-agnostic — LightGBM/TFT, the point-in-time feature pipeline, the evaluation panel) is orchestrated two ways that share the modelling and feature code unchanged:

- **Local MVP** — one command trains either model and backtests it honestly; a batch job writes monitored 28-day quantile forecasts for the whole catalogue to a prediction archive, with cold-start fallback and drift + forecast-error monitoring.
- **AWS-native platform** — the same code run on SageMaker as the full operations layer: Glue features, script-mode/BYOC training, Batch Transform, a continuous-training **Pipeline** with a WAPE promote-gate, CloudWatch monitoring wired to a retrain trigger, registry-based promote/rollback, and **CDK** infrastructure. See **[`aws/README.md`](aws/README.md)** — the whole track (Phases 0–9) is verified in a real account for ≈ $3.50.

**Status: complete** — portable MVP + AWS-native platform track both done and verified.

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
| 3. LightGBM baseline | Config-driven global model (per-quantile), MLflow tracking + registry, rolling-origin backtest | ✅ done |
| 4. Evaluation panel | WAPE/MASE/WRMSSE × hierarchy level × horizon; pinball loss + calibration report | ✅ done |
| 5. TFT upgrade | Darts TFT with native multi-horizon quantiles; honest comparison incl. retraining cost | ✅ done |
| 6. Batch inference + monitoring | Scheduled forecast job → prediction archive; data-drift + forecast-error monitoring; cold-start fallback | ✅ done |
| 7. Continuous Training | Scheduled + trigger-based automated retraining wired to the monitoring signal | ✅ AWS track |
| 8. Deployment harness | Shadow → A/B → promote → rollback on registry stages | ✅ AWS track |
| 9. Infrastructure as code | Storage, scheduler, registry, alarms defined as code | ✅ AWS track |

Phases 1–6 are the portable MVP: train either model with one command, backtest it honestly, and produce monitored 28-day quantile forecasts for the whole catalogue. **The platform layer (7–9) was delivered on the AWS-native track** rather than as a second local implementation — continuous training as a SageMaker Pipeline + EventBridge, shadow/A-B via the Model Registry, and infrastructure as **AWS CDK** (the portable-Terraform variant is a documented stretch). Everything AWS is in **[`aws/README.md`](aws/README.md)**.

### LightGBM vs TFT — the honest comparison (phase 5)

Both models were trained and backtested on the **same** California slice (120 series, 3 rolling-origin folds) and scored through the **same** evaluation panel, then compared on retraining cost and serving complexity — not accuracy alone.

| level | WAPE (LGBM) | WAPE (TFT) | MASE (LGBM) | MASE (TFT) |
|---|---|---|---|---|
| total | 0.351 | **0.311** | 2.81 | **2.49** |
| store | 0.369 | **0.343** | 1.61 | **1.50** |
| department | 0.496 | **0.488** | 1.14 | **1.12** |
| **item-store** (order level) | **0.764** | 0.764 | **0.931** | 0.933 |
| retrain time (3 folds) | **24 s** | 1550 s (~26 min) | — | — |

The TFT is modestly better at aggregate levels but **ties LightGBM at the item-store level — where replenishment orders are actually placed — for ~64× the retraining cost** and a heavier (torch-runtime) deployment. **The baseline stays the production model.** The transformer's edge may widen with a GPU, full history, and tuning (the AWS track); on this bounded, CPU-only evidence it does not earn its operational cost. Recommending the simpler model, with numbers behind it, is the intended outcome. This run is deliberately bounded (CA slice, 120 series, most-recent 2 years, small network, 5 epochs); full-scale TFT is deferred to the GPU track.

## AWS-native track

The same portable code, rebuilt as a managed platform on AWS and aligned to the **AWS ML Engineer – Associate (MLA-C01)** certification. The bridge is SageMaker **script mode** — AWS runs *our* training and feature code unchanged; only the orchestration moves to managed services. Full detail, run commands, and per-phase teardown are in **[`aws/README.md`](aws/README.md)**.

| Component | AWS service |
|---|---|
| Data lake · SQL exploration | S3 (Parquet) · Athena |
| Feature pipeline + data quality | Glue (PySpark) + Glue Data Quality · Feature Store (offline) |
| Training (LightGBM/TFT) | SageMaker training jobs — script mode + BYOC (CPU & GPU-Spot) |
| Tuning · built-in · bias | Automatic Model Tuning · DeepAR · Clarify |
| Experiments + versioning | SageMaker Experiments + Model Registry |
| Batch forecasting | SageMaker Batch Transform → prediction archive |
| Orchestration + CT | SageMaker Pipelines (WAPE promote-gate) + EventBridge (schedule + retrain trigger) |
| Monitoring | Processing job → CloudWatch custom metrics + alarms |
| Deployment | Registry approval status: shadow / A-B / promote / rollback |
| Infrastructure as code + CI/CD | AWS CDK (synth) + CodePipeline / CodeBuild |

**Four models compared on the same intermittent-demand data** (item-store WAPE): AMT-tuned LightGBM **0.6747** · DeepAR 0.6816 · baseline LightGBM 0.6963 · GPU-trained TFT 0.7011 — near-parity, the well-tuned tree ahead. The whole track (Phases 0–9) is verified in a real account for **≈ $3.50** with ephemeral compute only.

## Quickstart

```bash
poetry install

# Get the data (Kaggle account required; accept the competition rules first):
kaggle competitions download -c m5-forecasting-accuracy -p data/raw/
# unzip so that data/raw/ contains: sales_train_evaluation.csv, calendar.csv, sell_prices.csv

# Validate and convert to Parquet:
poetry run python -m demand_forecasting.data.convert --config config/config.yaml

# Build the point-in-time feature table:
poetry run python -m demand_forecasting.features.build --config config/config.yaml

# Train: rolling-origin backtest + final model, tracked and registered in MLflow:
poetry run python -m demand_forecasting.training.train --config config/config.yaml
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db   # inspect runs at http://localhost:5000

# Batch inference: 28-day quantile forecasts for the catalogue → prediction archive:
poetry run python -m demand_forecasting.inference.batch --config config/config.yaml

# Monitoring: operational health + forecast error + PSI data drift on the archive:
poetry run python -m demand_forecasting.monitoring.run --config config/config.yaml

# Optional — the TFT comparison (heavy deep-learning deps, not in CI/Docker):
poetry install --with tft
# set `training.model: tft` in config, then the same entry point runs the
# LightGBM-vs-TFT comparison on the CA slice → data/models/comparison/
poetry run python -m demand_forecasting.training.train --config config/config.yaml

# Optional — rich Evidently drift report:
poetry install --with monitoring

# Quality gate:
poetry run ruff check src/ tests/ && poetry run mypy src/ && poetry run pytest
```

`config/config.yaml` drives everything. `training.max_series` caps how many series train (keeps a laptop run to minutes; `0` = the full catalogue — a config change, not a code change).

## Repository structure

```
├── config/config.yaml         # all run parameters in one place
├── src/demand_forecasting/     # PORTABLE CORE (cloud-agnostic, shared by both tracks)
│   ├── data/                  # loading + schema validation + Parquet conversion
│   ├── features/              # point-in-time feature pipeline (shared)      [phase 2]
│   ├── training/              # LightGBM + TFT + MLflow                      [phases 3, 5]
│   ├── evaluation/            # metric panel + rolling-origin backtest       [phases 3, 4]
│   ├── inference/             # batch forecast job → prediction archive      [phase 6]
│   └── monitoring/            # drift + forecast-error checks                [phase 6]
├── aws/                        # AWS-NATIVE TRACK (see aws/README.md)         [phases 0–8]
│   ├── scripts/               # boto3/SDK launchers (run locally, submit to AWS)
│   ├── sagemaker/             # container entry points (training, eval, serve, monitor)
│   ├── glue/ · athena/ · docker/
│   └── README.md · PROGRESS.md
├── infra/cdk/                  # infrastructure as code — AWS CDK (synth-only) [phase 9]
├── buildspec.yml               # CI gate as a CodeBuild spec                  [phase 9]
├── tests/                     # unit + data + no-leakage + aws tests (synthetic fixtures)
└── notebooks/                 # EDA only, clearly separated
```

*(The platform tier lives in `aws/` rather than a local `src/…/platform/`: continuous training, deployment harness, and IaC were built cloud-natively — see the phase table above.)*

## Non-goals

Not chasing the M5 leaderboard; not heavy data cleaning; not ensembles or model exotica; not formal hierarchical reconciliation (a simple coherent roll-up is reported instead). The evaluation and operations rigor is the deliverable.

## Stack

**Core:** Python 3.12 · Poetry · LightGBM · Darts + PyTorch (TFT) · pydantic · MLflow · Evidently · pytest / ruff / mypy · Docker · GitHub Actions

**AWS track:** S3 · Athena · Glue (+ Data Quality) · SageMaker (Training, Processing, Experiments, Model Registry, Batch Transform, Automatic Model Tuning, DeepAR, Clarify, Pipelines, Feature Store) · ECR · CloudWatch · EventBridge · AWS CDK · CodePipeline / CodeBuild
