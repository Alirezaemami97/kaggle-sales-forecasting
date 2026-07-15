"""Prepare Athena/Glue-safe staging Parquet from the local processed data.

Two fixes over uploading data/processed/ directly:

1. The wide sales table (1,941 day columns) is unusable for SQL; Athena wants
   long rows. We reuse the repo's own `melt_sales`, so the local and cloud views
   of the data agree exactly, and write a compact Parquet ready to upload to
   `s3://<bucket>/raw/sales_long/`.
2. pandas/pyarrow writes datetime64[ns] columns as Parquet TIMESTAMP(NANOS),
   which Spark's Parquet reader (used by AWS Glue) cannot read at all — it
   fails at schema-inference time with "Illegal Parquet type: INT64
   (TIMESTAMP(NANOS,false))", before any column is even selected. Athena's
   engine (Trino) has no such issue, which is why Phase 1 never caught this.
   calendar's `date` column is never used in date arithmetic downstream (the
   integer `d` column is the real join/lag key), so we cast it to a plain
   'YYYY-MM-DD' string for the AWS copy — unambiguous for every reader. The
   local data/processed/calendar.parquet is untouched.

    python aws/scripts/prep_athena_data.py     # data/processed/ → data/aws_staging/
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from demand_forecasting.features.pipeline import melt_sales

logger = logging.getLogger(__name__)

# The columns Athena needs for the Phase-1 EDA queries (identity + day + units).
SALES_LONG_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales"]


def to_sales_long(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """Wide sales → long (one row per series-day), projected to the EDA columns."""
    long = melt_sales(sales_wide)
    return long[SALES_LONG_COLS]


def prepare_calendar(calendar: pd.DataFrame) -> pd.DataFrame:
    """Cast `date` to a plain 'YYYY-MM-DD' string — Spark's Parquet reader cannot
    read the nanosecond-precision timestamp pandas/pyarrow writes by default."""
    out = calendar.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def build(processed_dir: Path, out_dir: Path) -> Path:
    """Read the processed Parquet, apply the AWS-safe transforms, write staging."""
    out_dir.mkdir(parents=True, exist_ok=True)

    sales = pd.read_parquet(processed_dir / "sales.parquet")
    long = to_sales_long(sales)
    sales_out = out_dir / "sales_long.parquet"
    long.to_parquet(sales_out, index=False)
    logger.info("Wrote %d long rows to %s", len(long), sales_out)

    calendar = pd.read_parquet(processed_dir / "calendar.parquet")
    calendar_out = out_dir / "calendar.parquet"
    prepare_calendar(calendar).to_parquet(calendar_out, index=False)
    logger.info("Wrote Glue-safe calendar to %s", calendar_out)

    return sales_out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed", help="Where the Parquet lives")
    parser.add_argument("--out-dir", default="data/aws_staging", help="Where to write sales_long")
    args = parser.parse_args()
    build(Path(args.processed_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
