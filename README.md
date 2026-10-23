# Probabilistic Forecasting and Inventory Optimization

An end-to-end framework for converting probabilistic demand forecasts into inventory decisions.

The project uses the **M5 Forecasting Accuracy** dataset (and supports general retail panel datasets) to demonstrate a decision-centric pipeline:

1. Download and preprocess retail sales data.
2. Engineer time-series, calendar, event, and price features (including future-known covariates).
3. Train probabilistic forecasting models (**Poisson / Negative-Binomial GLMs** or **CatBoost Multi-Quantile Regression**).
4. Reconstruct discrete daily demand distributions and propagate uncertainty through stochastic lead times using **Numba-accelerated** discrete convolution and mixture algebra.
5. Solve a budget-constrained multi-product newsvendor problem per decision day.
6. Export forecasts, inventory decisions, and point/probabilistic forecasting metrics.

> **Project status:** Active development. The forecasting layer supports grouped Poisson/negative-binomial GLMs and CatBoost multi-quantile regression with non-parametric discrete distribution reconstruction. Lead-time and purchasing-cost inputs are synthetically generated for the M5 demonstration.

---

## Motivation

Many forecasting projects stop after producing point forecasts or reporting an accuracy metric. Inventory planning, however, depends on the **full demand distribution**, lead-time uncertainty, product economics, and operational constraints.

This project connects forecasting and optimization:

```text
Historical demand + Future covariates
                  ↓
    Probabilistic demand forecast
   (GLM count model or CatBoost quantiles)
                  ↓
      Daily discrete distributions
                  ↓
  Demand during lead time + review period
       (Numba random-sum mixture)
                  ↓
  Budget-constrained multi-product newsvendor
```

The goal is not merely to predict future demand, but to use uncertainty estimates to make an operational decision.

---

## Features

- Automatic download and MD5 checksum verification of the M5 dataset
- Memory-conscious preprocessing with Polars and store-level Parquet partitioning
- Synthetic lead-time, purchase-cost, and shortage-penalty generation
- Reusable discrete time-series panel builder (`DiscreteTimeSeries`) supporting both **backtesting** (`num_test_days > 0`) and **production inference** (`num_test_days = 0`)
- Lag, rolling-mean, and rolling-volatility demand features without train/test target leakage
- Calendar, event, and selling-price features, including **future-known covariates** (negative lags / leads for scheduled promotions, holidays, and prices beyond the last observed sales date)
- Composable forecaster abstractions (`ProbabilisticForecaster`, `GLMForecaster`, `CatBoostQuantileForecaster`)
- Grouped Poisson and negative-binomial regression with automatic overdispersion detection
- CatBoost `MultiQuantile` regression predicting a full grid of quantiles simultaneously
- Non-parametric quantile-to-discrete-PMF reconstruction with monotone rearrangement, linear interpolation, and exponential upper-tail extrapolation
- **Numba-accelerated** 1D convolution, random sums, quantile reconstruction, and $O(1)$-auxiliary-memory quantile lookup (`ddlt.py`)
- Per-day budget-constrained multi-product newsvendor optimization via Lagrangian binary search
- Store-level and category-level WMAPE, pinball loss, and empirical quantile coverage reporting
- Unit test suite (`pytest`)

---

## Repository Structure

```text
.
├── process_m5_dataset.py
├── run_m5_pipeline.py
├── time_series.py
├── forecasters.py
├── catboost_quantile.py
├── quantile_distribution.py
├── ddlt.py
├── discrete_distribution.py
├── probability_utils.py
├── multi_product_newsvendor.py
├── logging_utils.py
├── tests/
│   └── test_quantile_distribution.py
├── data/
│   ├── m5-forecasting-accuracy/
│   │   ├── calendar.csv
│   │   ├── sales_train_evaluation.csv
│   │   ├── sell_prices.csv
│   │   └── partitioned_data/
│   │       ├── calendar.parquet
│   │       ├── price.parquet
│   │       └── sales_data_<STORE_ID>.parquet
│   ├── synthetic_data/
│   │   ├── lead_time.parquet
│   │   └── newsvendor.parquet
│   └── output/
│       ├── model_data_<STORE_ID>.csv
│       ├── catboost_models_<STORE_ID>/
│       ├── feature_importance_catboost_<STORE_ID>.csv
│       ├── predictions_<STORE_ID>.csv
│       ├── predictions_catboost_<STORE_ID>.csv
│       ├── quantile_metrics_catboost.csv
│       ├── wmape_store.csv
│       ├── wmape_store_cat.csv
│       ├── wmape_store_catboost.csv
│       └── wmape_store_cat_catboost.csv
└── README.md
```

### Module Overview

#### `process_m5_dataset.py`

Handles data acquisition and preprocessing:

- downloads the M5 archive from Zenodo and verifies its MD5 checksum;
- extracts the raw CSV files;
- reshapes daily demand from wide to long format using Polars;
- joins calendar data with weekly selling prices;
- generates synthetic lead-time and newsvendor inputs;
- saves calendar, price, and store-partitioned sales tables as Parquet.

#### `time_series.py`

Contains the panel time-series and feature-engineering pipeline (`DiscreteTimeSeries`):

- constructs complete `ts_id × period` panels;
- creates leak-free train/test splits (or full-history panels when `num_test_days=0` in production);
- generates lag, rolling-mean, and rolling-standard-deviation features;
- adds calendar features (configurable weekend indicators or day-of-week encodings) and external features with historical lags, **future leads** (`lags=(-1, -2, ...)`), and rolling windows;
- optionally fits candidate historical demand distributions when lead time plus review cycle exceeds the forecast horizon;
- provides convenience delegation methods for training, prediction, evaluation, and DDLT calculation.

#### `forecasters.py`

Defines clean, composable forecaster abstractions:

- `ProbabilisticForecaster`: abstract base class defining `.fit()`, `.predict()`, and `.compute_ddlt()`;
- `GLMForecaster`: fits grouped Poisson / negative-binomial GLMs across forecast horizons and predicts horizon means/dispersions;
- `CatBoostQuantileForecaster`: fits multi-horizon CatBoost `MultiQuantile` models, predicts non-crossing quantile grids and distribution means, and propagates quantiles to DDLT distributions;
- standalone evaluation utilities (`compute_forecasting_metrics`, `evaluate_quantile_forecasts`).

#### `catboost_quantile.py`

Wraps CatBoost multi-quantile training, inference, and model persistence:

- builds `MultiQuantile:alpha=...` loss definitions;
- handles native categorical features and temporal validation splits for early stopping;
- provides `CatBoostQuantileModel` for saving/loading `.cbm` model bundles and extracting feature importances.

#### `quantile_distribution.py`

Converts predicted quantile vectors into discrete probability distributions over non-negative integer demand:

- removes quantile crossing via monotone rearrangement and clips at zero;
- interpolates linearly between predicted quantile knots;
- extrapolates the lower tail linearly to $\tau = 0$ and the upper tail exponentially through the last two knots up to $1 - \text{tol}$;
- discretizes via $P(D \le y) = P(Q(U) < y + 0.5)$ using Numba-compiled kernels.

#### `ddlt.py`

Implements **Numba-accelerated** distribution algebra and the model-agnostic `DDLTCalculator`:

- `_convolve_1d_numba` and `_random_sum_aligned_numba`: compiled 1D convolution and random-sum mixture kernels that accumulate directly into a 1D output vector without intermediate 2D matrix allocations;
- `_quantiles_to_pmf_numba` and `_quantiles_to_mean_batch_numba`: parallel JIT kernels for reconstructing PMFs and expectations from quantile matrices;
- `_pmf_quantile_scalar_numba` and `_batch_pmf_quantiles_numba`: $O(1)$-auxiliary-memory quantile lookup directly on PMFs without allocating or caching CDF arrays;
- `DDLTCalculator`: converts either multi-horizon quantile grids or GLM $(\mu, \alpha)$ parameters into lead-time demand distributions or service-level inventory targets.

#### `discrete_distribution.py`

Implements the `DiscreteDistribution` class and operations:

- creation from empirical observations (`from_data`) or parametric families (`from_parametric`: negative binomial, binomial, Poisson, geometric, discrete uniform);
- probability mass functions, cumulative distributions, and Numba-accelerated quantiles;
- shifting, bin merging, and bin splitting;
- convolution (`+`), identically distributed random sums (`@`), and non-identically distributed random sums (`random_sum`).

#### `multi_product_newsvendor.py`

Solves the multi-product newsvendor problem under a shared purchasing budget:

- for a fixed Lagrange multiplier $k \ge 0$, each product receives an adjusted critical fractile $\alpha_i(k)$ evaluated via `_batch_pmf_quantiles_numba`;
- a binary search finds the optimal multiplier $k^*$ satisfying the budget constraint;
- supports `group_cols` (e.g., `['store_id', 'day']`) to solve an independent budget-constrained problem on each decision day.

#### `run_m5_pipeline.py`

Runs the complete forecasting and inventory pipeline one store at a time, supporting both `--model glm` and `--model catboost`.

#### `logging_utils.py`

Configures structured, timestamped logging (`%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`) to stdout and optionally to a `.log` file via `setup_logging(log_file=..., level=...)`.

---

## Methodology

### 1a. Count-Regression GLMs (`--model glm`)

For each training group (`store_id`, `cat_id`, `overdispersion`) and forecast horizon $h \in \{0, \dots, 6\}$:

- a **Poisson GLM** is used when historical variance does not exceed the mean;
- a **Negative-Binomial GLM** is used when the series is overdispersed.

### 1b. CatBoost Multi-Quantile Regression (`--model catboost`)

For each forecast horizon $h \in \{0, \dots, 6\}$:

- a single store-level CatBoost model is trained with the `MultiQuantile` loss (summed pinball loss across all quantile levels $\tau \in \{0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99\}$);
- `item_id`, `dept_id`, `cat_id`, `wday`, and `event_type` are passed as native categorical features alongside lag, rolling mean, rolling standard deviation, month, and log selling price features;
- training rows whose future target $t + h$ falls in the test window are masked out (preventing look-ahead leakage), and the last 28 target days prior to the test window are used for early stopping;
- each predicted quantile vector is converted into a non-negative integer `DiscreteDistribution` via `quantile_distribution.py`.

### 2. Demand During Lead Time + Review Period (DDLT)

Daily predictive distributions $D_{t}, D_{t+1}, \dots$ are combined with an empirical store-level lead-time distribution $L$ shifted by the review cycle time $R$:

$$D_{t}^{L+R} = \sum_{j=0}^{L+R-1} D_{t+j}$$

Because $D_{t+j}$ are non-identically distributed across the forecast horizon and $L$ is stochastic, $D_{t}^{L+R}$ is a mixture of convolutions computed via `DDLTCalculator` and `DiscreteDistribution.random_sum`.

### 3. Multi-Product Newsvendor Under a Budget Constraint

On each decision day, each product $i$ has a DDLT distribution $F_i$, selling price $p_i$, purchase cost $c_i$, and shortage penalty $b_i$. The optimizer chooses order quantities $q_i$ satisfying:

$$\sum_{i} c_i q_i \le B, \qquad q_i(k) = F_i^{-1}\!\left(\operatorname{clip}\!\left(\frac{p_i - c_i(1 + k) + b_i}{p_i + b_i},\, 0,\, 1\right)\right)$$

via binary search over the shadow price $k \ge 0$.

---

## Requirements

Use Python **3.10 or later**.

Core dependencies:

```text
numpy
pandas
polars
scipy
statsmodels
matplotlib
requests
pyarrow
numba
catboost
pytest
```

---

## Installation

Clone the repository and create a virtual environment:

### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install numpy pandas polars scipy statsmodels matplotlib requests pyarrow numba catboost pytest
```

---

## How to Run

Run all commands from the repository root so local module imports resolve correctly.

### Step 1: Download and Process the M5 Dataset

```bash
python process_m5_dataset.py --download
```

To process a dataset that has already been downloaded:

```bash
python process_m5_dataset.py
```

Custom directories can be supplied:

```powershell
python process_m5_dataset.py --download --m5-dir data/m5-forecasting-accuracy --synthetic-dir data/synthetic_data --partitioned-dir data/m5-forecasting-accuracy/partitioned_data
```

### Step 2: Run Forecasting and Inventory Optimization

#### Run with GLMs (default)

```bash
python run_m5_pipeline.py --model glm --budget-per-store 15000
```

#### Run with CatBoost Multi-Quantile Regression

Full default configuration (1000 iterations, 730 days of training history, 12 quantiles):

```bash
python run_m5_pipeline.py --model catboost
```

Faster configuration for experimentation (e.g., 400 iterations, 365 days of training history):

```powershell
python run_m5_pipeline.py --model catboost --catboost-iterations 400 --catboost-train-days 365 --budget-per-store 15000
```

Custom quantile grid, output directory, and log file:

```powershell
python run_m5_pipeline.py --model catboost --catboost-iterations 400 --catboost-train-days 365 --quantiles 0.05 0.1 0.25 0.5 0.75 0.9 0.95 0.99 --out-dir data/output --log-file logs/m5_catboost.log
```

All CLI options for `run_m5_pipeline.py`:

| Flag | Default | Description |
|---|---|---|
| `--model` | `glm` | Forecasting model family (`glm` or `catboost`) |
| `--budget-per-store` | `15000` | Purchasing budget per store per decision day |
| `--quantiles` | `0.05 0.1 ... 0.95 0.99` | Quantile levels predicted by CatBoost |
| `--catboost-iterations` | `1000` | Max boosting iterations per horizon |
| `--catboost-train-days` | `730` | Rolling history window in days (`0` = full history) |
| `--synthetic-data-dir` | `data/synthetic_data` | Directory with `lead_time.parquet` and `newsvendor.parquet` |
| `--m5-data-dir` | `data/m5-forecasting-accuracy/partitioned_data` | Directory with partitioned M5 Parquet files |
| `--out-dir` | `data/output` | Directory for outputs |
| `--log-file` | `None` | Optional path to a `.log` file to save progress logs |
| `--log-level` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Step 3: Run the Unit Tests

```bash
python -m pytest tests -q
```

---

## Outputs

For each processed store:

- **Model artifacts:**
  - GLM: `data/output/model_data_<STORE_ID>.csv`
  - CatBoost: `data/output/catboost_models_<STORE_ID>/` (`.cbm` binaries + `catboost_meta.json`) and `data/output/feature_importance_catboost_<STORE_ID>.csv`
- **Predictions and inventory targets:**
  - `data/output/predictions_<STORE_ID>.csv` (GLM) or `data/output/predictions_catboost_<STORE_ID>.csv` (CatBoost), including horizon predictions, reconstructed means (`target_t_plus_<h>_mu`), quantile columns (`target_t_plus_<h>_q<tau>`), and `target_inventory`
- **Forecasting & calibration summaries:**
  - `data/output/wmape_store.csv` and `data/output/wmape_store_cat.csv` (or `*_catboost.csv`)
  - `data/output/quantile_metrics_catboost.csv` (pinball loss and empirical coverage per horizon and quantile level)

---

## Using the Components Independently

### 1. Reconstruct a Discrete Distribution from Predicted Quantiles

```python
import numpy as np
from quantile_distribution import DEFAULT_QUANTILES, quantiles_to_distribution

levels = np.array(DEFAULT_QUANTILES)
pred_quantiles = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0])

dist = quantiles_to_distribution(levels, pred_quantiles)
print("Support min/max:", dist.min, dist.max)
print("95th percentile:", dist.quantile(0.95)[0])
```

### 2. Compute a Numba-Accelerated Random Sum (DDLT)

```python
import numpy as np
from discrete_distribution import DiscreteDistribution

lead_time = DiscreteDistribution(pmf=np.array([0.2, 0.5, 0.3]), min_value=1)

daily_forecasts = [
    DiscreteDistribution.from_parametric("poisson", (3.0, 0)),
    DiscreteDistribution.from_parametric("poisson", (4.0, 0)),
    DiscreteDistribution.from_parametric("poisson", (5.0, 0)),
]

lead_time_demand = lead_time.random_sum(daily_forecasts)
print("DDLT 90th percentile:", lead_time_demand.quantile(0.90)[0])
```

### 3. Production Inference (`num_test_days=0`) with Future-Known Covariates

When deploying on a custom retail dataset where future sales actuals are unknown, set `num_test_days=0` and pass negative lags (`lags=(-1, -2, ...)`) to attach known future calendar/promotion/price features from a table that extends beyond the last observed sales date:

```python
from forecasters import CatBoostQuantileForecaster
from time_series import DiscreteTimeSeries, compute_lead_time_demand

ts = DiscreteTimeSeries(
    sales_df,
    ts_id_cols=["store_id", "item_id"],
    period_col="day",
    var_col="demand",
    lags=(1, 7, 14),
    rolling_windows=(7, 28),
    num_future_targets=7,
    num_test_days=0,                # Production mode: train on 100% of observed history
    use_one_hot_encoding=False,
    downsample_zeros_frac=None,
    fit_historical_distributions=False,
)

# Attach current and future-known covariates (e.g. promo_flag at t+1, t+2) from calendar_df:
ts.add_calendar_features(calendar_df, "wday", use_day_of_week_features=True)
ts.add_feature(calendar_df, "promo_flag", join_on=[], lags=(-1, -2))

forecaster = CatBoostQuantileForecaster(
    quantiles=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95),
    extra_cat_cols=("item_id",),
)
forecaster.fit(ts)

# Forecast and compute DDLT at the latest observed period (ts.end_period):
ddlt_predictions = compute_lead_time_demand(
    ts,
    forecaster,
    lead_time=lead_time_distr,
    lead_time_keys=["store_id"],
    cycle_time=1,
    return_distribution=True,
    pred_first_day=True,
)
```

---

## Development Roadmap

```text
Phase 1 — Forecasting foundation
  ✓ Time-series feature engineering (leak-free train/test & production mode)
  ✓ Poisson and negative-binomial regression
  ✓ Multi-horizon prediction
  ✓ Discrete predictive distributions

Phase 2 — Decision layer
  ✓ Numba-accelerated lead-time demand calculation
  ✓ Multi-product newsvendor with O(1)-memory quantile lookup
  ✓ Per-day shared budget constraint

Phase 3 — Model expansion & Architecture
  ✓ CatBoost multi-quantile regression
  ✓ Non-parametric quantile-to-distribution reconstruction
  ✓ Composable Forecaster and DDLTCalculator abstractions
  ✓ Future-known covariate leads for production inference
  ✓ Unit test suite

Phase 4 — Evaluation & Simulation
  □ Rolling-origin forecast backtesting
  □ Multi-period inventory replenishment simulation with carry-over stock
  □ Decision-based metrics (fill rate, realized profit, holding vs. shortage cost)
  □ GLM versus CatBoost decision benchmarking
```

---

## Disclaimer

This repository is intended for research, experimentation, and education.

The synthetic lead-time and economic inputs are not representative of any particular retailer. Inventory outputs should not be used as production recommendations without validating the data, assumptions, model calibration, and operational constraints.

---

## Acknowledgements

This project uses the M5 Forecasting Accuracy dataset, originally released for the M5 competition and hosted on Zenodo.
