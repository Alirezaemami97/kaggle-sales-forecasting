"""AWS Glue (PySpark) feature job — the cloud-native port of features/pipeline.py.

Reads the raw sales_long / calendar / prices Parquet from S3 (produced in Phase 1),
builds the SAME point-in-time features as the local pandas pipeline, and writes
Parquet to s3://<bucket>/features/. The logic mirrors the local pipeline so cloud
and local features agree row-for-row (that equality is the phase's verification).

This runs on managed Spark inside Glue, not locally — `awsglue`/`pyspark` are the
Glue runtime, so this file is never imported by the test suite.

Job parameters: --JOB_NAME (Glue-supplied), --bucket.
"""

import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F

LAGS = [7, 28]
ROLLING_WINDOWS = [7, 28]
# Calendar attributes carried into the feature table (known in advance at day t).
CALENDAR_KEEP = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                 "event_name_1", "event_type_1", "snap_ca", "snap_tx", "snap_wi"]


def load_raw(spark, bucket):
    sales = spark.read.parquet(f"s3://{bucket}/raw/sales_long/")
    calendar = spark.read.parquet(f"s3://{bucket}/raw/calendar/")
    prices = spark.read.parquet(f"s3://{bucket}/raw/prices/")
    # calendar `d` is 'd_1'.. — cast to the integer day index sales_long uses.
    calendar = calendar.withColumn("d", F.regexp_replace("d", "d_", "").cast("int"))
    return sales, calendar, prices


def join_calendar(sales, calendar):
    """Attach calendar attributes and collapse the three state SNAP flags into the
    one matching each row's own state."""
    out = sales.join(calendar.select(*CALENDAR_KEEP), on="d", how="left")
    out = out.withColumn(
        "snap",
        F.when(F.col("state_id") == "CA", F.col("snap_ca"))
        .when(F.col("state_id") == "TX", F.col("snap_tx"))
        .when(F.col("state_id") == "WI", F.col("snap_wi"))
        .otherwise(F.lit(0)),
    )
    return out.drop("snap_ca", "snap_tx", "snap_wi")


def add_price_features(df, prices):
    """Weekly price level, week-over-week change, and price relative to the item's
    department in the same store and week. Computed on the compact weekly table."""
    dept = F.regexp_extract("item_id", r"^(.*)_[^_]+$", 1)  # 'FOODS_3_090' -> 'FOODS_3'
    pr = prices.withColumn("dept_id", dept)

    w_item = Window.partitionBy("store_id", "item_id").orderBy("wm_yr_wk")
    prev = F.lag("sell_price").over(w_item)
    pr = pr.withColumn("price_pct_change", (F.col("sell_price") - prev) / prev)

    w_dept = Window.partitionBy("store_id", "dept_id", "wm_yr_wk")
    pr = pr.withColumn("price_rel_dept", F.col("sell_price") / F.avg("sell_price").over(w_dept))

    pr = pr.select("store_id", "item_id", "wm_yr_wk", "sell_price",
                   "price_pct_change", "price_rel_dept")
    return df.join(pr, on=["store_id", "item_id", "wm_yr_wk"], how="left")


def add_lags_and_rolling(df):
    """Lag features and trailing rolling mean/std ending at t-1 (point-in-time).

    The rolling window is `rowsBetween(-w, -1)` — the w days strictly before t —
    and is nulled when fewer than w days exist, matching the pandas min_periods=w.
    sales_long is a gap-free daily grid, so rows == days.
    """
    w = Window.partitionBy("id").orderBy("d")
    for lag in LAGS:
        df = df.withColumn(f"lag_{lag}", F.lag("sales", lag).over(w))
    for win in ROLLING_WINDOWS:
        trailing = w.rowsBetween(-win, -1)
        enough = F.count("sales").over(trailing) >= win
        df = df.withColumn(f"rolling_mean_{win}", F.when(enough, F.avg("sales").over(trailing)))
        df = df.withColumn(f"rolling_std_{win}", F.when(enough, F.stddev("sales").over(trailing)))
    return df


def build_features(spark, bucket):
    sales, calendar, prices = load_raw(spark, bucket)
    df = join_calendar(sales, calendar)
    df = add_price_features(df, prices)
    df = add_lags_and_rolling(df)
    # drop_pre_release: rows before an item's first listed price (NaN sell_price).
    return df.filter(F.col("sell_price").isNotNull())


def main():
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "bucket"])
    glue = GlueContext(SparkContext())
    job = Job(glue)
    job.init(args["JOB_NAME"], args)

    features = build_features(glue.spark_session, args["bucket"])
    features.write.mode("overwrite").parquet(f"s3://{args['bucket']}/features/")
    job.commit()


if __name__ == "__main__":
    main()
