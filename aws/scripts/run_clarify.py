"""Run a SageMaker Clarify PRE-TRAINING bias/data analysis on the sales sample.

Clarify has three modes with different prerequisites — exam material in itself:
  - pre-training bias  : data only, NO model            <- this script
  - post-training bias : needs model predictions
  - explainability/SHAP: needs a SERVABLE model (Clarify spins up a temporary
    shadow endpoint to query it) — deferred to Phase 5, where the inference
    handler is built for Batch Transform anyway. Deferred, not skipped.

Honest framing: pre-training bias metrics were designed for fairness facets
(protected attributes of people). This dataset is items, not people, so we use
the machinery to quantify DATASET IMBALANCE — does CA (4 stores vs 3/3 for
TX/WI) dominate the training signal? CI (class imbalance between facet groups)
and DPL (difference in the share of "sold something" days) answer exactly that.
The label is precomputed as binary `sold = sales > 0` — unambiguous, and it
matches the 68%-zeros intermittency framing used since the local EDA.

    python aws/scripts/run_clarify.py --bucket <name> --role-arn <role>
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# boto3/sagemaker imported lazily in main(); build_clarify_frame is CI-tested.

STAGING = Path("data/aws_staging")
CLARIFY_COLUMNS = ["sold", "state_id", "dept_id", "store_id"]
FACET = "state_id"
FACET_VALUE = "CA"


def build_clarify_frame(sales: pd.DataFrame) -> pd.DataFrame:
    """Clarify input: binary label + categorical facets, nothing else.

    `state_id` is derived from the store prefix (CA_1 -> CA) because the DeepAR
    sample loader does not carry it. Column order matters — Clarify's DataConfig
    headers must match the CSV exactly. Pure — unit-tested in CI.
    """
    frame = pd.DataFrame(
        {
            "sold": (sales["sales"] > 0).astype(int),
            "state_id": sales["store_id"].astype(str).str.split("_").str[0],
            "dept_id": sales["dept_id"].astype(str),
            "store_id": sales["store_id"].astype(str),
        }
    )
    return frame[CLARIFY_COLUMNS]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--instance-type", default="ml.m5.xlarge")
    parser.add_argument(
        "--max-series", type=int, default=300,
        help="Distribution statistics stabilise quickly; 300 series is plenty",
    )
    args = parser.parse_args()

    import boto3
    import sagemaker
    import sagemaker_compat
    from prep_deepar_data import load_sample
    from sagemaker import clarify

    sagemaker_compat.apply()  # Windows SDK infinite-loop fix; see sagemaker_compat.py

    sales = load_sample(STAGING / "sales_long.parquet", args.max_series, seed=42)
    frame = build_clarify_frame(sales)
    logger.info(
        "Clarify input: %d rows; sold share overall=%.3f, %s=%s share=%.3f",
        len(frame),
        frame["sold"].mean(),
        FACET,
        FACET_VALUE,
        (frame[FACET] == FACET_VALUE).mean(),
    )
    local_csv = STAGING / "clarify_input.csv"
    frame.to_csv(local_csv, index=False)

    boto_session = boto3.Session()
    session = sagemaker.Session(boto_session=boto_session)
    s3_input = f"s3://{args.bucket}/clarify/input/clarify_input.csv"
    # No trailing slash: Clarify joins paths with "/" itself, and a double slash
    # in an S3 key fails the job at the data-download step.
    s3_output = f"s3://{args.bucket}/clarify/output"
    boto_session.client("s3").upload_file(
        str(local_csv), args.bucket, "clarify/input/clarify_input.csv"
    )

    processor = clarify.SageMakerClarifyProcessor(
        role=args.role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        sagemaker_session=session,
    )
    data_config = clarify.DataConfig(
        s3_data_input_path=s3_input,
        s3_output_path=s3_output,
        label="sold",
        headers=CLARIFY_COLUMNS,
        dataset_type="text/csv",
    )
    bias_config = clarify.BiasConfig(
        label_values_or_threshold=[1],       # positive outcome: the item sold
        facet_name=FACET,
        facet_values_or_threshold=[FACET_VALUE],  # CA vs everyone else
    )
    processor.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=bias_config,
        # Explicit list rather than "all": CDDL needs a conditioning "group
        # variable" we do not have, and requesting it just bakes a permanent
        # error entry into the report.
        methods=["CI", "DPL", "KL", "JS", "KS", "LP", "TVD"],
        wait=True,
    )
    logger.info("Clarify report: %s/analysis.json (plus report.html/pdf)", s3_output)


if __name__ == "__main__":
    main()
