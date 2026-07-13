"""Tests for the data layer: schema validation, loaders, and Parquet conversion."""

from pathlib import Path

import pandas as pd
import pytest

from demand_forecasting.config import Config, DataConfig, FeaturesConfig, PathsConfig
from demand_forecasting.data.convert import convert_all
from demand_forecasting.data.loader import load_calendar, load_prices, load_sales
from demand_forecasting.data.schema import (
    validate_calendar,
    validate_prices,
    validate_sales,
)


class TestValidateSales:
    def test_valid_data_passes(self, sales_df: pd.DataFrame) -> None:
        stats = validate_sales(sales_df)
        assert stats.n_series == 3
        assert stats.n_days == 10
        assert 0.0 < stats.zero_rate < 1.0

    def test_missing_id_column_fails(self, sales_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing required columns"):
            validate_sales(sales_df.drop(columns=["store_id"]))

    def test_negative_sales_fail(self, sales_df: pd.DataFrame) -> None:
        sales_df.loc[0, "d_3"] = -1
        with pytest.raises(ValueError, match="negative unit sales"):
            validate_sales(sales_df)

    def test_gap_in_day_columns_fails(self, sales_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not contiguous"):
            validate_sales(sales_df.drop(columns=["d_5"]))

    def test_null_in_day_columns_fails(self, sales_df: pd.DataFrame) -> None:
        sales_df.loc[1, "d_2"] = None
        with pytest.raises(ValueError, match="null values in day columns"):
            validate_sales(sales_df)

    def test_duplicate_series_id_fails(self, sales_df: pd.DataFrame) -> None:
        sales_df.loc[2, "id"] = sales_df.loc[0, "id"]
        with pytest.raises(ValueError, match="duplicated series ids"):
            validate_sales(sales_df)


class TestValidateCalendar:
    def test_valid_data_passes(self, calendar_df: pd.DataFrame) -> None:
        stats = validate_calendar(calendar_df)
        assert stats.n_days == 10
        assert stats.first_date == "2011-01-29"

    def test_date_gap_fails(self, calendar_df: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not contiguous"):
            validate_calendar(calendar_df.drop(index=3))

    def test_bad_snap_flag_fails(self, calendar_df: pd.DataFrame) -> None:
        calendar_df.loc[0, "snap_CA"] = 2
        with pytest.raises(ValueError, match="snap_CA"):
            validate_calendar(calendar_df)


class TestValidatePrices:
    def test_valid_data_passes(self, prices_df: pd.DataFrame) -> None:
        stats = validate_prices(prices_df)
        assert stats.n_rows == 5
        assert stats.n_items == 2

    def test_non_positive_price_fails(self, prices_df: pd.DataFrame) -> None:
        prices_df.loc[0, "sell_price"] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            validate_prices(prices_df)

    def test_duplicate_key_fails(self, prices_df: pd.DataFrame) -> None:
        dupe = pd.concat([prices_df, prices_df.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicated"):
            validate_prices(dupe)


def _write_raw_csvs(
    raw_dir: Path,
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> None:
    raw_dir.mkdir(parents=True)
    sales_df.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)
    calendar_df.to_csv(raw_dir / "calendar.csv", index=False)
    prices_df.to_csv(raw_dir / "sell_prices.csv", index=False)


def test_loaders_downcast_dtypes(
    tmp_path: Path,
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> None:
    _write_raw_csvs(tmp_path / "raw", sales_df, calendar_df, prices_df)

    sales, _ = load_sales(tmp_path / "raw" / "sales_train_evaluation.csv")
    assert str(sales["d_1"].dtype) == "int16"
    assert str(sales["item_id"].dtype) == "category"

    calendar, _ = load_calendar(tmp_path / "raw" / "calendar.csv")
    assert str(calendar["snap_CA"].dtype) == "int8"
    assert str(calendar["date"].dtype).startswith("datetime64")

    prices, _ = load_prices(tmp_path / "raw" / "sell_prices.csv")
    assert str(prices["sell_price"].dtype) == "float32"


def test_convert_all_writes_parquet(
    tmp_path: Path,
    sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> None:
    _write_raw_csvs(tmp_path / "raw", sales_df, calendar_df, prices_df)
    config = Config(
        project_name="test",
        random_seed=42,
        paths=PathsConfig(
            raw_dir=tmp_path / "raw",
            processed_dir=tmp_path / "processed",
            models_dir=tmp_path / "models",
            forecasts_dir=tmp_path / "forecasts",
        ),
        data=DataConfig(
            sales_file="sales_train_evaluation.csv",
            calendar_file="calendar.csv",
            prices_file="sell_prices.csv",
            subsample_series=2,
        ),
        features=FeaturesConfig(lags=[2, 3], rolling_windows=[3], drop_pre_release=True),
    )

    convert_all(config)

    sales = pd.read_parquet(tmp_path / "processed" / "sales.parquet")
    assert len(sales) == 2  # subsample_series honoured
    assert str(sales["d_1"].dtype) == "int16"  # dtypes survive the Parquet roundtrip
    assert (tmp_path / "processed" / "calendar.parquet").exists()
    assert (tmp_path / "processed" / "prices.parquet").exists()
