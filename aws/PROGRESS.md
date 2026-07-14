# AWS-native track — progress tracker

One file, all ten phases (0–9): what each phase is, why it exists, its status, and what's verified. Update the status line as each phase completes. See `aws/README.md` for the service map and cost-discipline rules; see `docs/AWS_DEPLOYMENT.md` (gitignored, local-only) for the full phase-by-phase build spec this tracker summarizes.

**Legend:** ✅ done and verified · 🔜 next up · ⬜ not started

---

## Phase 0 — Account hardening ✅

**What & why:** one-time Console setup so the rest of the track can't accidentally overspend: root MFA, a daily IAM user (not root), Region locked to `us-east-1`, AWS Budgets alerts, Cost Explorer, and a `project` cost-allocation tag. This is the safety net every later phase depends on.

**Status:** done. IAM user configured with AWS CLI credentials (`aws sts get-caller-identity` verified).

**Checklist** (re-check occasionally, especially after long gaps):
- [ ] AWS Budgets — monthly cost budget with alerts at $10 / $25 / $50 / $75
- [ ] Cost Explorer enabled
- [ ] CloudWatch billing alerts on
- [ ] `project` cost-allocation tag activated (Billing → Cost allocation tags)
- [ ] Root MFA on; daily work done as an IAM user with MFA, not root
- [ ] Region locked to us-east-1
- [ ] Credits confirmed to cover S3, Athena, Glue, SageMaker

---

## Phase 1 — Data into S3, queryable with Athena ✅

**What & why:** the data lake every later phase reads from. The wide M5 sales table (1,941 day-columns) is unusable for SQL, so we reuse the repo's own `melt_sales` to produce a long-format table, upload everything as Parquet (cheap, column-pruned Athena scans), and register it as Athena external tables — schema-on-read, no data duplicated.

**Built (committed `f0e8460`):**
- `aws/scripts/s3_setup.py` — idempotent bucket + `raw/features/models/forecasts/athena-results` prefixes, SSE-AES256 encryption, public-access block, `project` tag
- `aws/scripts/prep_athena_data.py` — wide → long sales table via `melt_sales`
- `aws/scripts/upload_data.py` — uploads sales_long/calendar/prices Parquet to `raw/`
- `aws/athena/create_tables.sql` + `eda_queries.sql` — external tables + SQL EDA

**Verified in your account** (bucket `demand-forecasting-alireza-2026`, results annotated in `eda_queries.sql`):
- 30,490 series × 1,941 days = 59,181,090 rows
- **zero_sales_share = 0.6800** — matches the local M2 EDA (~0.68), confirming the cloud data is correct
- State volume: CA 29.2M > TX 19.2M > WI 18.5M; top store CA_3 (11.4M units); FOODS dominates category volume (45.9M); FOODS_3 dominates department (32.9M)

**Cost:** storage only (~$0.10/month); Athena scans were megabytes (fractions of a cent).

---

## Phase 2 — Glue join + Data Quality + Feature Store 🔜

**What & why:** the AWS-native, point-in-time-correct feature pipeline — the cloud replacement for the local `features/pipeline.py`. A **Glue job** (PySpark) joins sales+calendar+prices and builds the same lag/rolling/calendar/price features; **Glue Data Quality** rules replace the pydantic schema checks (no nulls in keys, sales ≥ 0, contiguous dates); features land in **SageMaker Feature Store** (offline store) keyed by an event-time field, which is the managed answer to the leakage problem.

**Not started.** Depends on Phase 1's `raw/` data (done).

**Cost estimate:** Glue ≈ $0.44/DPU-hour, a small job is a few DPU-minutes (<$1/run). Feature Store **offline only** — never enable the online store (it bills hourly).

---

## Phase 3 — Train the baseline as a SageMaker training job ⬜

**What & why:** the bridge phase — SageMaker **script mode** runs the repo's existing `train.py` unchanged on managed compute (`ml.m5.xlarge`, CPU), logs to **Experiments**, and registers the result in **Model Registry**. Proves the portable code ports without modification.

**Cost estimate:** ~$0.23/hour, a training run is minutes (cents).

---

## Phase 4 — Evaluation + tuning + model comparison ⬜

**What & why:** the M4 evaluation panel and M5 LightGBM-vs-TFT comparison, redone on AWS: a **Processing job** for the metric panel, **Automatic Model Tuning** (capped at 6–8 jobs), **DeepAR** (SageMaker's built-in time-series model) as a third comparison point, and **TFT** on a GPU **Spot** instance (short runs, on a subset).

**Cost estimate:** the one phase with real spend — AMT + GPU ≈ $10–20. Cap AMT jobs; verify GPU instances stop.

---

## Phase 5 — Batch inference ⬜

**What & why:** cloud-native version of the local M6 batch job. **SageMaker Batch Transform** scores the catalogue and writes 28-day quantile forecasts to `forecasts/` in S3 — no idle endpoint, so this is one of the cheapest phases.

**Cost estimate:** ephemeral, cents per run.

---

## Phase 6 — Orchestrate with SageMaker Pipelines ⬜

**What & why:** one DAG (data → train → evaluate → register → batch), triggered on an **EventBridge** schedule (e.g. weekly) — the AWS-native scheduled Continuous Training. A condition step only registers/promotes a model if it beats the incumbent on WAPE.

**Cost estimate:** Pipelines itself is free; pay only for the underlying ephemeral steps.

---

## Phase 7 — Monitoring + Continuous Training trigger ⬜

**What & why:** closes the loop, cloud-native. **SageMaker Model Monitor** watches the batch outputs for data/model-quality drift, a **CloudWatch alarm** fires on forecast-error or drift breaching a threshold, and **EventBridge + Step Functions** wire that alarm to a retrain. CloudTrail gives the audit trail.

**Cost estimate:** ephemeral processing jobs + pennies of CloudWatch/EventBridge/CloudTrail.

---

## Phase 8 — Deployment harness (shadow + A/B) ⬜

**What & why:** cloud-native version of shadow → A/B → promote/rollback. Candidate model runs in Batch Transform alongside the incumbent (shadow), the catalogue is split to compare on real actuals (A/B), and the winner is promoted to the `production` stage in Model Registry (rollback = one Registry change).

**Cost estimate:** ephemeral batch jobs, cents. Avoid a real-time endpoint unless demoing; delete same day if used.

---

## Phase 9 — Infrastructure as code with CDK + CI/CD ⬜

**What & why:** everything built by hand in the Console, re-expressed as an **AWS CDK** app (Python) and deployed via `cdk deploy` — proof the whole system is reproducible from code. **CodePipeline + CodeBuild** run the ruff/mypy/pytest gate and redeploy on push. `cdk destroy` is the final, one-command teardown.

**Cost estimate:** CDK/CloudFormation free; CodeBuild ~$0.005/build-minute.

---

## Overall cost so far

| Phase | Est. cost | Actual |
|---|---|---|
| 0 | $0 | $0 |
| 1 | <$1 | storage + cents of Athena scans |

Check Cost Explorer periodically — it should read a few dollars at most through Phase 3.
