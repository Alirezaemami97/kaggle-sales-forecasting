-- Phase-1 EDA in SQL — the Athena analogue of the local M2 EDA notebook.
-- Run these after create_tables.sql. They scan only the sales_long columns they
-- reference (Parquet column pruning), so each costs a fraction of a cent.

-- 1. Scale of the problem: how many series and days.
SELECT count(DISTINCT id) AS n_series,
       count(DISTINCT d)  AS n_days,
       count(*)           AS n_rows
FROM demand_forecasting.sales_long;
-- Result:
-- #	n_series	n_days	n_rows
-- 1	30490	1941	59181090


-- 2. Intermittency: share of series-days with zero sales (expect ~0.68, the
--    reason we use WAPE/MASE over MAPE). Should match the local EDA.
SELECT avg(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END) AS zero_sales_share
FROM demand_forecasting.sales_long;
-- Result:
-- #	zero_sales_share
-- 1	0.6799776584040612

-- 3. Total units by state, then by store — the top of the hierarchy.
SELECT state_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY state_id
ORDER BY total_units DESC;
-- Result:
-- #	state_id	total_units
-- 1	CA	29196717
-- 2	TX	19228405
-- 3	WI	18502051

SELECT store_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY store_id
ORDER BY total_units DESC;
-- Result:
-- #	store_id	total_units
-- 1	CA_3	11363540
-- 2	CA_1	7832248
-- 3	TX_2	7329642
-- 4	WI_2	6697988
-- 5	WI_3	6542557
-- 6	TX_3	6205940
-- 7	CA_2	5818395
-- 8	TX_1	5692823
-- 9	WI_1	5261506
-- 10	CA_4	4182534


-- 4. Volume by category and department (where the demand concentrates).
SELECT cat_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY cat_id
ORDER BY total_units DESC;
-- Result:
-- #	cat_id	total_units
-- 1	FOODS	45922427
-- 2	HOUSEHOLD	14764090
-- 3	HOBBIES	6240656

SELECT dept_id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY dept_id
ORDER BY total_units DESC;
-- Result:
-- #	dept_id	total_units
-- 1	FOODS_3	32937002
-- 2	HOUSEHOLD_1	11722853
-- 3	FOODS_2	7795025
-- 4	HOBBIES_1	5699014
-- 5	FOODS_1	5190400
-- 6	HOUSEHOLD_2	3041237
-- 7	HOBBIES_2	541642


-- 5. Top 20 item-store series by total units — the head of a long tail.
SELECT id, sum(sales) AS total_units
FROM demand_forecasting.sales_long
GROUP BY id
ORDER BY total_units DESC
LIMIT 20;
-- Result:
-- #	id	total_units
-- 1	FOODS_3_090_CA_3_evaluation	253859
-- 2	FOODS_3_586_TX_2_evaluation	195120
-- 3	FOODS_3_586_TX_3_evaluation	151862
-- 4	FOODS_3_586_CA_3_evaluation	136269
-- 5	FOODS_3_090_CA_1_evaluation	128855
-- 6	FOODS_3_090_WI_3_evaluation	123500
-- 7	FOODS_3_090_TX_2_evaluation	121275
-- 8	FOODS_3_090_TX_3_evaluation	116773
-- 9	FOODS_3_252_TX_2_evaluation	115613
-- 10	FOODS_3_586_TX_1_evaluation	114010
-- 11	FOODS_3_226_WI_3_evaluation	99578
-- 12	FOODS_3_555_TX_2_evaluation	98708
-- 13	FOODS_3_090_TX_1_evaluation	95051
-- 14	FOODS_3_120_CA_3_evaluation	90412
-- 15	FOODS_3_586_CA_1_evaluation	88846
-- 16	FOODS_3_252_TX_3_evaluation	87642
-- 17	FOODS_3_586_WI_3_evaluation	86999
-- 18	FOODS_3_694_WI_3_evaluation	86863
-- 19	FOODS_3_252_CA_3_evaluation	82861
-- 20	FOODS_3_541_CA_3_evaluation	80495
