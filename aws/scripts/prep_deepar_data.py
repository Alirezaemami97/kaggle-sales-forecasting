"""Convert the long sales table to DeepAR JSON Lines and upload train/test.

DeepAR is a SageMaker BUILT-IN algorithm: AWS's code in AWS's container, no
entry point of ours — which is why the data must be in ITS format. One JSON
line per series:

    {"start": "2011-01-29", "target": [3, 0, 1, ...], "cat": [store_idx, dept_idx]}

Decisions that keep the comparison honest:
  - Same 3000-series sample as the LightGBM runs — selected with the shared
    `cap_series` and the same seed, so all Phase-4 models see the same series.
  - Leading zeros are trimmed (each series starts at its first positive sale):
    M5 series nominally start at day 1, but most items were not on sale yet —
    the same reality drop_pre_release handles in the feature pipeline. Years of
    pre-release zeros would teach the model "this item never sells".
  - Train targets stop `prediction_length` days before the end; the test file
    carries the full series. DeepAR itself scores the trailing window of each
    test series (test:mean_wQuantileLoss, test:RMSE) — built-in evaluation.

    python aws/scripts/prep_deepar_data.py --bucket <name>
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# boto3 imported lazily in main(); the converters are pure and CI-tested.

M5_EPOCH = pd.Timestamp("2011-01-29")  # day d=1
PREDICTION_LENGTH = 28
STAGING = Path("data/aws_staging")


def series_to_json(group: pd.DataFrame, cat: list[int], holdout: int) -> str | None:
    """One series -> one JSON line, trimmed to start at its first positive sale.

    `holdout` > 0 drops that many trailing days (the train file); 0 keeps the
    full series (the test file). Returns None for series with no sales at all
    or too little history to both train and score.
    """
    g = group.sort_values("d")
    nonzero = g[g["sales"] > 0]
    if nonzero.empty:
        return None
    g = g[g["d"] >= int(nonzero["d"].iloc[0])]
    if holdout:
        g = g.iloc[:-holdout]
    # Need at least one point beyond the scored window to learn from.
    if len(g) <= PREDICTION_LENGTH:
        return None
    start = M5_EPOCH + pd.Timedelta(days=int(g["d"].iloc[0]) - 1)
    return json.dumps(
        {
            "start": start.strftime("%Y-%m-%d"),
            "target": [int(v) for v in g["sales"]],
            "cat": cat,
        }
    )


def build_jsonl(sales: pd.DataFrame) -> tuple[list[str], list[str]]:
    """(train_lines, test_lines) for the sampled sales frame. Pure — CI-tested.

    `cat` carries store and department as stable integer codes; DeepAR learns
    an embedding per category, its only view of the hierarchy.
    """
    store_codes = {v: i for i, v in enumerate(sorted(sales["store_id"].astype(str).unique()))}
    dept_codes = {v: i for i, v in enumerate(sorted(sales["dept_id"].astype(str).unique()))}

    train_lines, test_lines = [], []
    for _, group in sales.groupby("id", observed=True, sort=True):
        cat = [
            store_codes[str(group["store_id"].iloc[0])],
            dept_codes[str(group["dept_id"].iloc[0])],
        ]
        train = series_to_json(group, cat, holdout=PREDICTION_LENGTH)
        test = series_to_json(group, cat, holdout=0)
        if train is not None and test is not None:
            train_lines.append(train)
            test_lines.append(test)
    return train_lines, test_lines


def load_sample(sales_long_path: Path, max_series: int, seed: int) -> pd.DataFrame:
    """The same pushdown-and-cap pattern as train_entry.load_features, on the
    raw sales_long table (DeepAR wants raw demand, not engineered features)."""
    import pyarrow.dataset as ds

    from demand_forecasting.training.dataset import cap_series

    dataset = ds.dataset(str(sales_long_path), format="parquet")
    ids = dataset.to_table(columns=["id"]).column("id").unique().to_pylist()
    unique_ids = pd.DataFrame({"id": sorted(str(i) for i in ids)})
    keep = cap_series(unique_ids, max_series, seed)["id"].tolist()
    table = dataset.to_table(
        columns=["id", "store_id", "dept_id", "d", "sales"],
        filter=ds.field("id").isin(keep),
    )
    return table.to_pandas()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--sales-long", default=str(STAGING / "sales_long.parquet"))
    parser.add_argument("--max-series", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42, help="Must match config.random_seed")
    parser.add_argument(
        "--skip-upload", action="store_true", help="Build the local files only (validation)"
    )
    args = parser.parse_args()

    sales = load_sample(Path(args.sales_long), args.max_series, args.seed)
    logger.info("Sampled %d rows, %d series", len(sales), sales["id"].nunique())
    train_lines, test_lines = build_jsonl(sales)
    logger.info("Built %d train / %d test series", len(train_lines), len(test_lines))

    out_dir = STAGING / "deepar"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.json").write_text("\n".join(train_lines), encoding="utf-8")
    (out_dir / "test.json").write_text("\n".join(test_lines), encoding="utf-8")
    logger.info("Wrote %s and %s", out_dir / "train.json", out_dir / "test.json")
    if args.skip_upload:
        return

    import boto3

    s3 = boto3.client("s3")
    s3.upload_file(str(out_dir / "train.json"), args.bucket, "deepar/train/train.json")
    s3.upload_file(str(out_dir / "test.json"), args.bucket, "deepar/test/test.json")
    logger.info("Uploaded to s3://%s/deepar/{train,test}/", args.bucket)


if __name__ == "__main__":
    main()
