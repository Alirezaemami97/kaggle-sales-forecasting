-- Full feature table (all 25 columns the Phase-2 Glue job writes), superseding
-- the ad hoc 6-column features_check preview table used to verify row counts.
-- Glue Data Quality (aws/glue/data_quality_rules.dqdl) evaluates against this
-- table, and it's generically useful for any future Athena exploration.
-- Replace <BUCKET> with your bucket name before running.

CREATE EXTERNAL TABLE IF NOT EXISTS demand_forecasting.features (
  id                string,
  item_id           string,
  dept_id           string,
  cat_id            string,
  store_id          string,
  state_id          string,
  d                 int,
  sales             int,
  date              string,
  wm_yr_wk          int,
  wday              int,
  month             int,
  year              int,
  event_name_1      string,
  event_type_1      string,
  snap              int,
  sell_price        float,
  price_pct_change  float,
  price_rel_dept    float,
  lag_7             float,
  lag_28            float,
  rolling_mean_7    float,
  rolling_std_7     float,
  rolling_mean_28   float,
  rolling_std_28    float
)
STORED AS PARQUET
LOCATION 's3://<BUCKET>/features/';
