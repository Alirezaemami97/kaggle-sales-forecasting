-- Athena external tables over the Parquet uploaded to s3://<BUCKET>/raw/.
-- Replace <BUCKET> with your bucket name before running.
-- Set the Athena query-results location once (Settings → Manage) to
--   s3://<BUCKET>/athena-results/  — Athena writes every result there.
--
-- External tables store no data: they describe how to read the Parquet in place,
-- so dropping a table never deletes data. Athena bills per byte scanned, and
-- Parquet + column pruning keeps these EDA scans in the megabytes.

CREATE DATABASE IF NOT EXISTS demand_forecasting;

-- Long-format sales: one row per series-day (the melt_sales output). Primary EDA table.
CREATE EXTERNAL TABLE IF NOT EXISTS demand_forecasting.sales_long (
  id        string,
  item_id   string,
  dept_id   string,
  cat_id    string,
  store_id  string,
  state_id  string,
  d         int,
  sales     int
)
STORED AS PARQUET
LOCATION 's3://<BUCKET>/raw/sales_long/';

-- Calendar attributes keyed by day id (d = 'd_1' .. 'd_1969').
-- date is a plain 'YYYY-MM-DD' string, not a timestamp: Spark's Parquet reader
-- (used by the Phase-2 Glue job) can't read the nanosecond timestamps pandas
-- writes, and the calendar's real join/lag key is the integer `d`, not `date`.
CREATE EXTERNAL TABLE IF NOT EXISTS demand_forecasting.calendar (
  date          string,
  wm_yr_wk      int,
  weekday       string,
  wday          int,
  month         int,
  year          int,
  d             string,
  event_name_1  string,
  event_type_1  string,
  event_name_2  string,
  event_type_2  string,
  snap_ca       int,
  snap_tx       int,
  snap_wi       int
)
STORED AS PARQUET
LOCATION 's3://<BUCKET>/raw/calendar/';

-- Weekly sell prices per (store, item, week).
CREATE EXTERNAL TABLE IF NOT EXISTS demand_forecasting.prices (
  store_id    string,
  item_id     string,
  wm_yr_wk    int,
  sell_price  float
)
STORED AS PARQUET
LOCATION 's3://<BUCKET>/raw/prices/';
