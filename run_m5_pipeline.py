import argparse
import gc
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasters import CatBoostQuantileForecaster, GLMForecaster, evaluate_quantile_forecasts
from logging_utils import setup_logging
from multi_product_newsvendor import _solve_multi_product_newsvendor
from quantile_distribution import DEFAULT_QUANTILES
from time_series import DiscreteTimeSeries, compute_distribution_dict, compute_lead_time_demand

logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_DATA_DIR = Path("data/synthetic_data")
DEFAULT_M5_DATA_DIR = Path("data/m5-forecasting-accuracy/partitioned_data")
DEFAULT_OUT_DIR = Path("data/output")
DEFAULT_BUDGET_PER_STORE = 15_000.0
DEFAULT_MODEL = "glm"
DEFAULT_CATBOOST_ITERATIONS = 1000
DEFAULT_CATBOOST_TRAIN_DAYS = 730


def parse_args():
    parser = argparse.ArgumentParser(description="Run the M5 forecasting and inventory pipeline.")
    parser.add_argument(
        "--synthetic-data-dir",
        type=Path,
        default=DEFAULT_SYNTHETIC_DATA_DIR,
        help=f"Directory containing synthetic input data (default: {DEFAULT_SYNTHETIC_DATA_DIR}).",
    )
    parser.add_argument(
        "--m5-data-dir",
        type=Path,
        default=DEFAULT_M5_DATA_DIR,
        help=f"Directory containing partitioned M5 data (default: {DEFAULT_M5_DATA_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for pipeline outputs (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--budget-per-store",
        type=float,
        default=DEFAULT_BUDGET_PER_STORE,
        help=f"Inventory budget available per store per decision day (default: {DEFAULT_BUDGET_PER_STORE:g}).",
    )
    parser.add_argument(
        "--model",
        choices=["glm", "catboost"],
        default=DEFAULT_MODEL,
        help=f"Forecasting model: Poisson/negative-binomial GLMs or CatBoost multi-quantile regression (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUANTILES),
        help="Quantile levels predicted by the CatBoost model.",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=DEFAULT_CATBOOST_ITERATIONS,
        help=f"Maximum number of boosting iterations per horizon (default: {DEFAULT_CATBOOST_ITERATIONS}).",
    )
    parser.add_argument(
        "--catboost-train-days",
        type=int,
        default=DEFAULT_CATBOOST_TRAIN_DAYS,
        help=f"Days of history used to train CatBoost; use 0 for the full history (default: {DEFAULT_CATBOOST_TRAIN_DAYS}).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional path to a .log file to save pipeline logs (in addition to console output).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity level (default: INFO).",
    )
    return parser.parse_args()


def build_time_series(sales_df, calendar_df, prices_df, model, fit_historical_distributions=True):
    """Builds the feature set. CatBoost uses native categorical features and rolling volatility features."""
    is_catboost = model == "catboost"
    ts = DiscreteTimeSeries(
        sales_df,
        ts_id_cols=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        period_col='day',
        var_col='demand',
        num_future_targets=7,
        num_test_days=28,
        include_std_dev=is_catboost,
        use_one_hot_encoding=not is_catboost,
        fit_historical_distributions=fit_historical_distributions,
    )
    ts.add_calendar_features(calendar_df, 'wday')
    ts.add_feature(prices_df, 'sell_price', join_on=['store_id', 'item_id'])
    ts.add_feature(
        calendar_df.fillna({'event_type': 'no_event'}),
        'event_type',
        join_on=[],
        is_cat_feature=True,
        default_value='no_event',
    )
    if is_catboost:
        ts.add_feature(calendar_df, 'month', join_on=[])
    return ts


def main(
    synthetic_data_dir,
    m5_data_dir,
    out_dir,
    budget_per_store,
    model="glm",
    quantiles=DEFAULT_QUANTILES,
    catboost_iterations=DEFAULT_CATBOOST_ITERATIONS,
    catboost_train_days=DEFAULT_CATBOOST_TRAIN_DAYS,
    log_file=None,
    log_level="INFO",
):
    setup_logging(log_file=log_file, level=log_level)
    pipeline_t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    # GLM outputs keep their original file names; other models get a suffix.
    suffix = "" if model == "glm" else f"_{model}"

    logger.info("Loading calendar, price, lead-time, and newsvendor tables...")
    calendar_df = pd.read_parquet(m5_data_dir / 'calendar.parquet')
    prices_df = pd.read_parquet(m5_data_dir / 'price.parquet')
    prices_df['sell_price'] = np.log1p(prices_df['sell_price'])

    wmape_store = []
    wmape_store_cat = []
    quantile_metrics = []

    lead_time_df = pd.read_parquet(synthetic_data_dir / 'lead_time.parquet')
    lead_time_distr, lead_time_keys = compute_distribution_dict(lead_time_df, value='lead_time', keys=['store_id'])
    del lead_time_df; gc.collect()

    newsvendor_df = pd.read_parquet(synthetic_data_dir / 'newsvendor.parquet')
    cycle_time = 1
    horizon = 7

    def _compute_wmape(df):
        df['wmape'] = np.round(100 * df['abs_diff'] / df['actual'], 2)
        df['under_wmape'] = np.round(100 * df['under_diff'] / df['actual'], 2)
        df['over_wmape'] = np.round(100 * df['over_diff'] / df['actual'], 2)

    store_files = sorted(p for p in m5_data_dir.iterdir() if p.name.startswith('sales_data'))
    for store_idx, item in enumerate(store_files, start=1):
        store_t0 = time.perf_counter()
        sales_df = pd.read_parquet(item)
        store_name = re.search(r"sales_data_([^.]+)\.parquet", item.name).group(1)
        logger.info(
            "=== [%d/%d] Processing store %s (model=%s, budget=%.2f/day) ===",
            store_idx,
            len(store_files),
            store_name,
            model,
            budget_per_store,
        )

        needs_fallback_distr = (lead_time_distr[store_name].max + cycle_time) > horizon
        ts = build_time_series(
            sales_df, calendar_df, prices_df, model, fit_historical_distributions=needs_fallback_distr
        )
        del sales_df; gc.collect()

        logger.info("Null values in sell_price column: %d", int(ts.df['sell_price'].isna().sum()))
        if model == "glm":
            forecaster = GLMForecaster(train_groups=['store_id', 'cat_id'])
            model_data = forecaster.fit(ts)
            model_path = out_dir / f'model_data_{store_name}.csv'
            model_data.to_csv(model_path, index=False)
            logger.info("Saved GLM model metadata to %s", model_path)
        else:
            forecaster = CatBoostQuantileForecaster(
                quantiles=quantiles,
                max_train_days=catboost_train_days or None,
                catboost_params={'iterations': catboost_iterations},
            )
            model_data = forecaster.fit(ts)
            models_dir = out_dir / f'catboost_models_{store_name}'
            model_data.save(models_dir)
            fi_path = out_dir / f'feature_importance{suffix}_{store_name}.csv'
            model_data.feature_importance().to_csv(fi_path, index=False)
            logger.info("Saved CatBoost models to %s and feature importances to %s", models_dir, fi_path)

        predictions = compute_lead_time_demand(
            ts,
            forecaster,
            lead_time_distr,
            lead_time_keys,
            cycle_time=cycle_time,
            service_levels=None,
            model_type=model,
            return_distribution=True,
            pred_first_day=False,
        )
        if model == "catboost":
            quantile_metrics.append(
                evaluate_quantile_forecasts(
                    predictions, ts.target_cols, model_data.quantiles, group_cols=['store_id', 'cat_id']
                )
            )
        del ts; gc.collect()

        # Solve newsvendor model independently on each test day (Option A):
        store_newsvendor_df = newsvendor_df
        if 'sell_price' in predictions.columns:
            store_newsvendor_df = newsvendor_df.drop(columns='sell_price')
            predictions['sell_price'] = np.expm1(predictions['sell_price'])  # Keep original sell price

        predictions = predictions.merge(store_newsvendor_df, on=['store_id', 'item_id'], how='left')
        predictions = _solve_multi_product_newsvendor(
            predictions, budget_per_store, group_cols=['store_id', 'day']
        )
        pred_path = out_dir / f'predictions{suffix}_{store_name}.csv'
        predictions.to_csv(pred_path, index=False)
        logger.info("Saved predictions and inventory decisions to %s", pred_path)

        # Compute Forecasting Metrics:
        df_wmape_store_cat = predictions.groupby(['store_id', 'cat_id'], as_index=False, observed=True)[
            ['pred', 'actual', 'under_diff', 'over_diff', 'abs_diff']
        ].sum()
        df_wmape_store = df_wmape_store_cat.groupby('store_id', as_index=False, observed=True)[
            ['pred', 'actual', 'under_diff', 'over_diff', 'abs_diff']
        ].sum()
        _compute_wmape(df_wmape_store_cat)
        _compute_wmape(df_wmape_store)
        logger.info(
            "Store %s WMAPE=%.2f%% (under=%.2f%%, over=%.2f%%) in %.2fs",
            store_name,
            float(df_wmape_store['wmape'].iloc[0]),
            float(df_wmape_store['under_wmape'].iloc[0]),
            float(df_wmape_store['over_wmape'].iloc[0]),
            time.perf_counter() - store_t0,
        )
        wmape_store_cat.append(df_wmape_store_cat)
        wmape_store.append(df_wmape_store)
        del predictions; gc.collect()
        break

    wmape_store_cat = pd.concat(wmape_store_cat)
    wmape_store = pd.concat(wmape_store)
    wmape_store.to_csv(out_dir / f'wmape_store{suffix}.csv', index=False)
    wmape_store_cat.to_csv(out_dir / f'wmape_store_cat{suffix}.csv', index=False)
    if quantile_metrics:
        quantile_metrics = pd.concat(quantile_metrics)
        qm_path = out_dir / f'quantile_metrics{suffix}.csv'
        quantile_metrics.to_csv(qm_path, index=False)
        summary_str = quantile_metrics.groupby('quantile')[['pinball_loss', 'coverage']].mean().to_string()
        logger.info("Quantile evaluation summary (saved to %s):\n%s", qm_path, summary_str)
    logger.info("Pipeline finished in %.2fs", time.perf_counter() - pipeline_t0)


if __name__ == '__main__':
    args = parse_args()
    main(
        synthetic_data_dir=args.synthetic_data_dir,
        m5_data_dir=args.m5_data_dir,
        out_dir=args.out_dir,
        budget_per_store=args.budget_per_store,
        model=args.model,
        quantiles=tuple(sorted(args.quantiles)),
        catboost_iterations=args.catboost_iterations,
        catboost_train_days=args.catboost_train_days,
        log_file=args.log_file,
        log_level=args.log_level,
    )
