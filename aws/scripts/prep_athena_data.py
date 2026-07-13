"""Prepare a slim, long-format sales table for Athena SQL / EDA.

The wide sales table (1,941 day columns) is unusable for SQL; Athena wants long
rows. We reuse the repo's own `melt_sales`, so the local and cloud views of the
data agree exactly, and write a compact Parquet ready to upload to
`s3://<bucket>/raw/sales_long/`.

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


def build(processed_dir: Path, out_dir: Path) -> Path:
    """Read the wide sales Parquet, melt to long, write sales_long.parquet."""
    sales = pd.read_parquet(processed_dir / "sales.parquet")
    long = to_sales_long(sales)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sales_long.parquet"
    long.to_parquet(out, index=False)
    logger.info("Wrote %d long rows to %s", len(long), out)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed", help="Where the Parquet lives")
    parser.add_argument("--out-dir", default="data/aws_staging", help="Where to write sales_long")
    args = parser.parse_args()
    build(Path(args.processed_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
