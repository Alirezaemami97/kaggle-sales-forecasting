# AWS-native deployment track

The portable system (in `src/`) is cloud-agnostic — LightGBM/TFT, MLflow, Evidently, the point-in-time feature pipeline. This `aws/` subtree rebuilds the **operations layer** on AWS managed services, keeping the modelling code unchanged: the bridge is **SageMaker script mode**, so AWS runs *our* training and feature code as-is. The training entry point (`aws/sagemaker/train_entry.py`) imports the portable package unmodified and talks to AWS only through the container contract (`SM_CHANNEL_*`, `SM_MODEL_DIR`).

The track is built phase by phase, each a working increment with its own teardown. It doubles as hands-on preparation for the **AWS ML Engineer – Associate (MLA-C01)** certification.

## Local ↔ cloud split

| Stays in the repo (portable) | Moves to AWS (the platform) |
|---|---|
| feature pipeline, model code, evaluation panel, tests | storage, training compute, orchestration |
| the honest engineering narrative | registry, batch serving, monitoring, IaC |

## Service map (target)

| Component | AWS service | Portable equivalent |
|---|---|---|
| Data lake | S3 | local `data/` |
| Join + data quality | Glue + Glue Data Quality | pandas + pydantic |
| SQL exploration | Athena | pandas |
| Feature store (point-in-time) | SageMaker Feature Store (offline) | Parquet + shared pipeline |
| Training image | ECR (custom image, `aws/docker/Dockerfile`) | local Poetry env |
| Training (LightGBM/TFT) | SageMaker training jobs (script mode) | local training |
| Experiments + versioning | SageMaker Experiments + Model Registry | MLflow |
| Batch forecasting | SageMaker Batch Transform | local batch job |
| Orchestration | SageMaker Pipelines + EventBridge/Step Functions | Python + schedule |
| Monitoring | Model Monitor + CloudWatch | Evidently + PSI |
| Infrastructure as code | **AWS CDK** | Terraform |

## Cost discipline (the four habits)

You pay for **running** resources, not for "having a project". The whole build is ~$20–35 with discipline, on a ~$100 credit budget.

1. Check Cost Explorer at the **start and end** of every session.
2. Keep compute **ephemeral** (training / processing / batch jobs auto-stop). **Avoid real-time endpoints**; if you must, use Serverless and delete it the same day.
3. **Shut down SageMaker Studio apps** after every session; never leave a NAT Gateway or stuck job running.
4. **Cap** Automatic Model Tuning (6–8 jobs) and keep **GPU** runs short, few, on a subset, on **Spot**.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Account hardening (MFA, IAM, Budgets, Region lock) | ✅ done |
| 1 | Data into S3 as Parquet; queryable with Athena | ✅ done |
| 2 | Glue join + Data Quality + Feature Store | ✅ done |
| 3 | Train LightGBM as a SageMaker training job | ✅ done |
| 4 | Evaluation + tuning + DeepAR + TFT (GPU Spot) | 🔜 next |
| 5 | Batch Transform → forecasts archive | ⬜ |
| 6 | SageMaker Pipeline + EventBridge schedule | ⬜ |
| 7 | Model Monitor + CloudWatch + retrain trigger | ⬜ |
| 8 | Shadow + A/B deployment harness | ⬜ |
| 9 | CDK (all infra as code) + CodePipeline CI/CD | ⬜ |

## Phase 1 — run it

Install the optional AWS tooling and configure credentials first:

```bash
poetry install --with aws
aws configure                 # IAM user access keys, region us-east-1, output json
aws sts get-caller-identity   # should print your IAM user

# 1. Create + configure the bucket (idempotent):
python aws/scripts/s3_setup.py --bucket demand-forecasting-<your-suffix>

# 2. Build the long-format sales table and upload the three datasets:
python aws/scripts/prep_athena_data.py
python aws/scripts/upload_data.py --bucket demand-forecasting-<your-suffix>
```

Then in the **Athena console** (set the results location to `s3://<bucket>/athena-results/` once): run `aws/athena/create_tables.sql`, then the queries in `aws/athena/eda_queries.sql`. The zero-sales share should read ~0.68, matching the local EDA. Nothing here runs persistent compute, so there is nothing to tear down beyond the (cheap) S3 storage.

## Phase 3 — run it

Trains the LightGBM baseline on SageMaker from the Phase 2 feature output, then registers it. Needs a SageMaker execution role and a running Docker daemon.

```bash
# 1. Build the training image and push it to ECR (~1.4GB; first push is the slow one):
python aws/scripts/build_and_push_image.py --tag latest

# 2. Train on managed compute (ephemeral; the instance destroys itself at exit):
python aws/scripts/run_sagemaker_training.py --bucket demand-forecasting-<your-suffix> \
    --role-arn arn:aws:iam::<account>:role/demand-forecasting-sagemaker-role --max-series 3000

# 3. Register the artifact it prints as version 1 (status: PendingManualApproval):
python aws/scripts/register_model.py --role-arn arn:aws:iam::<account>:role/demand-forecasting-sagemaker-role \
    --model-data s3://demand-forecasting-<your-suffix>/models/<job-name>/output/model.tar.gz
```

A capped run is ~140 billable seconds on `ml.m5.xlarge` (about a cent). The model is registered **pending, not approved**: approval means "beat the incumbent on WAPE", which is Phase 4's evaluation and Phase 6's condition step — training succeeding is not a reason to promote. Teardown: nothing runs persistently; delete the ECR repository (~$0.10/GB/month) when done with the track.

**Why a custom image rather than a prebuilt one:** AWS's framework containers pin their own Python (the newest SKLearn training image is 3.9) and this project targets 3.12. Rather than contort the portable package to fit a container, `aws/docker/Dockerfile` owns the runtime — `python:3.12-slim` plus the `sagemaker-training` toolkit, which supplies the same script-mode contract, so `train_entry.py` is identical either way.
