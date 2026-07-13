-- Phase-1 EDA in SQL — the Athena analogue of the local M2 EDA notebook.
-- Run these after create_tables.sql. They scan only the sales_long columns they
-- reference (Parquet column pruning), so each costs a fraction of a cent.

-- 1. Scale of the problem: how many series and days.
SELECT count(DISTINCT id) AS n_series,
       count(DISTINCT d)  AS n_days,
       count(*)           AS n_rows
FROM demand_forecasting.sales_long;

-- 2. Intermittency: share of series-days with zero sales (expect ~0.68, the
--    reason we use WAPE/MASE over MAPE). Should match the local EDA.
SELECT avg(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END) AS zero_sales_share
FROM demand_forecasting.sales_long;

-- 3. Total units by state, then by store — the top of the hierarchy.
SELECT state_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY state_id
ORDER BY total_units DESC;

SELECT store_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY store_id
ORDER BY total_units DESC;

-- 4. Volume by category and department (where the demand concentrates).
SELECT cat_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY cat_id
ORDER BY total_units DESC;

SELECT dept_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY dept_id
ORDER BY total_units DESC;

-- 5. Top 20 item-store series by total units — the head of a long tail.
SELECT id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY id
ORDER BY total_units DESC
LIMIT 20;
