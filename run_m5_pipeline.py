import argparse
import re
import gc
from pathlib import Path

import numpy as np
import pandas as pd

from time_series import DiscreteTimeSeries, compute_lead_time_demand, compute_distribution_dict
from multi_product_newsvendor import _solve_multi_product_newsvendor

DEFAULT_SYNTHETIC_DATA_DIR = Path("data/synthetic_data")
DEFAULT_M5_DATA_DIR = Path("data/m5-forecasting-accuracy/partitioned_data")
DEFAULT_OUT_DIR = Path("data/output")
DEFAULT_BUDGET_PER_STORE = 15_000.0


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
        help=f"Inventory budget available per store (default: {DEFAULT_BUDGET_PER_STORE:g}).",
    )
    return parser.parse_args()


def main(synthetic_data_dir, m5_data_dir, out_dir, budget_per_store):
    out_dir.mkdir(parents=True, exist_ok=True)
    calendar_df = pd.read_parquet(m5_data_dir / 'calendar.parquet')
    prices_df = pd.read_parquet(m5_data_dir / 'price.parquet')
    prices_df['sell_price'] = np.log1p(prices_df['sell_price'])
    
    metrics = {'wmape_store': [], 'wmape_store_cat': []}
    
    lead_time_df = pd.read_parquet(synthetic_data_dir / 'lead_time.parquet')
    lead_time_distr, lead_time_keys = compute_distribution_dict(lead_time_df, value='lead_time', keys=['store_id'])
    del lead_time_df; gc.collect()
    
    newsvendor_df = pd.read_parquet(synthetic_data_dir / 'newsvendor.parquet')

    def _compute_wmape(df):
        df['wmape'] = np.round(100 * df['abs_diff'] / df['actual'], 2)
        df['under_wmape'] = np.round(100 * df['under_diff'] / df['actual'], 2)
        df['over_wmape'] = np.round(100 * df['over_diff'] / df['actual'], 2)
        
    for item in m5_data_dir.iterdir():
        if not item.name.startswith('sales_data'):
            continue
        sales_df = pd.read_parquet(item)
        store_name = re.search(r"sales_data_([^.]+)\.parquet", item.name).group(1)
        sales_df = sales_df[sales_df['cat_id']=='FOODS']
        print(f"Training Forecasting Model for store: {store_name}")
        ts = DiscreteTimeSeries(sales_df, ts_id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], period_col = 'day', var_col = 'demand', num_future_targets = 7, num_test_days=28)
        del sales_df; gc.collect()

        ts.add_calendar_features(calendar_df, day_of_week_col='wday')
        ts.add_feature(prices_df, 'sell_price', join_on=['store_id', 'item_id'])
        ts.add_feature(calendar_df.fillna({'event_type': 'no_event'}), 'event_type', join_on=[], is_cat_feature=True, default_value='no_event')
        
        print(f"null values in price column: {ts.df['sell_price'].isna().sum()}")
        model_data = ts.train_glm(train_groups=['store_id', 'cat_id'])
        model_data.to_csv(out_dir / f'model_data_{store_name}.csv', index=False)

        predictions = compute_lead_time_demand(ts, model_data, lead_time_distr, lead_time_keys, cycle_time=1, service_levels=None, model_type='glm', return_distribution=True)
        del ts; gc.collect()
        
        # Solve newsvendor model:
        print(f"Solving newsvendor model for store: {store_name}")
        if 'sell_price' in predictions.columns:
            newsvendor_df.drop(columns='sell_price', inplace=True)
            predictions['sell_price'] = np.expm1(predictions['sell_price']) # Keep original sell price
        
        predictions = predictions.merge(newsvendor_df, on=['store_id', 'item_id'], how='left')
        predictions = _solve_multi_product_newsvendor(predictions, budget_per_store)
        # predictions = predictions.groupby(['store_id', 'day'], as_index=False)[predictions.columns].apply(lambda x: _solve_multi_product_newsvendor(x, BUDGET_PER_STORE)).reset_index(drop=True)
        predictions.to_csv(out_dir / f'predictions_{store_name}.csv', index=False)

        # Compute Forecasting Metrics:
        wmape_store_cat = predictions.groupby(['store_id', 'cat_id'], as_index=False)[['pred', 'actual', 'under_diff', 'over_diff', 'abs_diff']].sum()
        wmape_store = wmape_store_cat.groupby('store_id', as_index=False)[['pred', 'actual', 'under_diff', 'over_diff', 'abs_diff']].sum()
        _compute_wmape(wmape_store_cat)
        _compute_wmape(wmape_store)
        metrics['wmape_store_cat'].append(wmape_store_cat)
        metrics['wmape_store'].append(wmape_store)
    
    metrics['wmape_store_cat'] = pd.concat(metrics['wmape_store_cat'])
    metrics['wmape_store'] = pd.concat(metrics['wmape_store'])
    metrics['wmape_store'].to_csv(out_dir / 'wmape_store.csv', index=False)
    metrics['wmape_store_cat'].to_csv(out_dir / 'wmape_store_cat.csv', index=False)

if __name__ == '__main__':
    args = parse_args()
    main(
        synthetic_data_dir=args.synthetic_data_dir,
        m5_data_dir=args.m5_data_dir,
        out_dir=args.out_dir,
        budget_per_store=args.budget_per_store,
    )
    
