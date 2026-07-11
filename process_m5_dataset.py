from __future__ import annotations
import argparse
import gc
import numpy as np
import polars as pl
import requests
import shutil
import hashlib
from pathlib import Path

M5_URL = (
    "https://zenodo.org/records/12636070/files/"
    "m5-forecasting-accuracy.zip?download=1"
)
M5_MD5 = "86f57416a314197f40a17cc6fc60cbb4"

def calculate_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the MD5 checksum of a file."""
    checksum = hashlib.md5()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            checksum.update(chunk)

    return checksum.hexdigest()

def download_m5_dataset(
    data_dir: Path,
    *,
    force_download: bool = False,
) -> Path:
    """Download, verify, and extract the M5 dataset."""
    archive_path = data_dir.parent / f"{data_dir.name}.zip"
    partial_path = archive_path.with_suffix(".zip.part")

    data_dir.parent.mkdir(parents=True, exist_ok=True)

    # Avoid downloading again when a valid archive already exists.
    if archive_path.exists() and not force_download:
        if calculate_md5(archive_path) == M5_MD5:
            print("M5 archive already downloaded.")
        else:
            print("Existing archive failed checksum verification. Downloading again.")
            archive_path.unlink()
    else:
        archive_path.unlink(missing_ok=True)

    if not archive_path.exists():
        # Remove any incomplete download left by an earlier run.
        partial_path.unlink(missing_ok=True)

        try:
            with requests.get(
                M5_URL,
                stream=True,
                timeout=(10, 120),  # connection timeout, read timeout
            ) as response:
                response.raise_for_status()

                with partial_path.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)

            downloaded_md5 = calculate_md5(partial_path)

            if downloaded_md5 != M5_MD5:
                raise ValueError(
                    "M5 archive checksum verification failed: "
                    f"expected {M5_MD5}, received {downloaded_md5}"
                )

            # Atomic rename: the final path appears only after a complete download.
            partial_path.replace(archive_path)

        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

        print("M5 dataset download complete.")

    if data_dir.exists() and not force_download:
        print("M5 dataset is already extracted.")
        return data_dir

    if data_dir.exists():
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(archive_path, data_dir)

    print(f"M5 dataset extracted to {data_dir}")
    return data_dir

def load_dataset(m5_dir: Path) -> pl.DataFrame:
    path_to_sales_train = m5_dir / 'sales_train_evaluation.csv'
    ts_id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']    
    return (
        pl.scan_csv(path_to_sales_train)
        .unpivot(index=ts_id_cols, variable_name='day', value_name='demand')
        .with_columns(pl.col('day').str.split('_').list.get(1).cast(pl.Int64))
        .collect()
    )

def load_calendar_and_prices(m5_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    path_to_calendar = m5_dir / 'calendar.csv'
    path_to_sell_prices = m5_dir / 'sell_prices.csv'
    
    calendar_lazy = pl.scan_csv(path_to_calendar).select(
        pl.col('d').str.split('_').list.get(1).cast(pl.Int64).alias('day'),
        pl.col('event_name_1').alias('event_name'),
        pl.col('event_type_1').alias('event_type'),
        pl.col('wday', 'month', 'year', 'wm_yr_wk')
    )
    
    prices_lazy = pl.scan_csv(path_to_sell_prices)
    prices_with_calendar = (
        prices_lazy
        .join(calendar_lazy, on='wm_yr_wk', how='inner')
        .drop('wm_yr_wk')
    )
    return (
        calendar_lazy.drop('wm_yr_wk').collect(), 
        prices_with_calendar.collect()
    )

def generate_synthetic_lead_times(
    dataset: pl.DataFrame, 
    seed: int = 0, 
    num_shipments: int = 1000
) -> pl.DataFrame:
    
    rng = np.random.default_rng(seed)
    stores = dataset['store_id'].unique().to_numpy()
    n_stores = len(stores)
    
    min_lts = rng.integers(1, 3, size=n_stores)
    max_lts = rng.integers(4, 7, size=n_stores)
    
    store_col = np.repeat(stores, num_shipments)
    repeated_mins = np.repeat(min_lts, num_shipments)
    repeated_maxs = np.repeat(max_lts, num_shipments)
    
    consignment_col = np.tile(np.arange(1, num_shipments + 1), n_stores)
    lead_time_col = rng.integers(repeated_mins, repeated_maxs)
    
    return pl.DataFrame({
        'store_id': store_col,
        'consignment_id': consignment_col,
        'lead_time': lead_time_col
    })

def generate_synthetic_newsvendor_data(price_df: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    
    filtered_df = (
        price_df
        .filter(pl.col('day') == pl.col('day').max().over(['item_id', 'store_id']))
        .select(['item_id', 'store_id', 'sell_price'])
    )
    
    sell_prices = filtered_df['sell_price'].to_numpy()
    
    result_df = filtered_df.with_columns(
        purchase_cost=pl.Series(
            rng.uniform(0.3 * sell_prices, 0.8 * sell_prices)
        ).round(2),
        shortage_penalty=(pl.col('sell_price') * 0.25)
    )
    
    return result_df

def process_m5_data(m5_dir: Path, synthetic_dir: Path, partitioned_dir: Path):
    print("Loading raw datasets...")
    dataset = load_dataset(m5_dir)
    calendar_df, price_df = load_calendar_and_prices(m5_dir)
    
    # Generate and save synthetic data:
    print("Generating synthetic data...")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Lead Time:
    lead_time_df = generate_synthetic_lead_times(dataset)
    lead_time_file_path = synthetic_dir / 'lead_time.parquet'
    lead_time_df.write_parquet(lead_time_file_path)
    print(f"Lead Time data saved to: {lead_time_file_path}")
    
    # Cleanup lead_time memory
    del lead_time_df; gc.collect()
    
    # 2. Newsvendor Problem Data:
    newsvendor_df = generate_synthetic_newsvendor_data(price_df)
    newsvendor_file_path = synthetic_dir / 'newsvendor.parquet'
    newsvendor_df.write_parquet(newsvendor_file_path)
    print(f"Newsvendor data saved to: {newsvendor_file_path}")
    
    # Cleanup newsvendor memory
    del newsvendor_df; gc.collect()
    
    # Save the calendar and price data:
    print("Saving calendar and price data...")
    partitioned_dir.mkdir(parents=True, exist_ok=True)
    
    calendar_file_path = partitioned_dir / 'calendar.parquet'
    calendar_df.write_parquet(calendar_file_path)
    print(f"Calendar data saved to: {calendar_file_path}")

    price_file_path = partitioned_dir / 'price.parquet'
    price_df.write_parquet(price_file_path)
    print(f"Price data saved to: {price_file_path}")
    
    # Cleanup calendar and price memory before partitioning large datasets
    del calendar_df, price_df; gc.collect()
    
    # Partition and store the sales data:
    print("Partitioning sales data...")
    partitions_dict = dataset.partition_by('store_id', as_dict=True)
    
    # Delete the massive unpartitioned dataset before iterating/writing
    del dataset; gc.collect()
    
    for store_id, df in partitions_dict.items():
        # Store_id tuple handling (Polars partitions return a tuple for group keys)
        store_str = store_id[0] if isinstance(store_id, tuple) else store_id 
        file_path = partitioned_dir / f'sales_data_{store_str}.parquet'
        df.write_parquet(file_path)
        print(f"Sales data for store: {store_str} saved to: {file_path}")
    
    # Final cleanup
    del partitions_dict; gc.collect()
    print("Processing complete.")

def main():
    parser = argparse.ArgumentParser(description="M5 Dataset Processor and Synthetic Data Generator")
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Download and extract the M5 dataset if it does not already exist."
    )
    parser.add_argument(
        "--m5-dir", 
        type=Path, 
        default=Path("data/m5-forecasting-accuracy"), 
        help="Path to the main M5 data directory."
    )
    parser.add_argument(
        "--synthetic-dir", 
        type=Path, 
        default=Path("data/synthetic_data"), 
        help="Directory to save generated synthetic data."
    )
    parser.add_argument(
        "--partitioned-dir", 
        type=Path, 
        default=None, 
        help="Directory to save partitioned sales data. Defaults to '<m5-dir>/partitioned_data'."
    )
    
    args = parser.parse_args()
    
    # Resolve paths based on arguments
    m5_dir = args.m5_dir
    synthetic_dir = args.synthetic_dir
    partitioned_dir = args.partitioned_dir if args.partitioned_dir else (m5_dir / "partitioned_data")

    if args.download:
        print(f"Ensuring M5 dataset is downloaded to {m5_dir}...")
        download_m5_dataset(m5_dir)
        
    process_m5_data(
        m5_dir=m5_dir, 
        synthetic_dir=synthetic_dir, 
        partitioned_dir=partitioned_dir
    )

if __name__ == "__main__":
    main()