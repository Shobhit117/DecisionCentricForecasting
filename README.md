# Probabilistic Forecasting and Inventory Optimization

An end-to-end framework for converting probabilistic demand forecasts into inventory decisions.

The project uses the **M5 Forecasting Accuracy** dataset to demonstrate a decision-centric pipeline:

1. Download and preprocess retail sales data.
2. Engineer time-series, calendar, event, and price features.
3. Train count-based probabilistic forecasting models.
4. Construct demand-during-lead-time distributions.
5. Solve a budget-constrained multi-product newsvendor problem.
6. Export forecasts, inventory decisions, and forecasting metrics.

> **Project status:** Experimental / work in progress. The current implementation uses Poisson and negative-binomial generalized linear models. Lead-time and purchasing-cost inputs are synthetically generated for demonstration.

---

## Motivation

Many forecasting projects stop after producing point forecasts or reporting an accuracy metric. Inventory planning, however, depends on the **full demand distribution**, lead-time uncertainty, product economics, and operational constraints.

This project connects forecasting and optimization:

```text
Historical demand
      ↓
Probabilistic demand forecast
      ↓
Demand during lead time
      ↓
Economic loss model
      ↓
Budget-constrained inventory decision
```

The goal is not merely to predict future demand, but to use uncertainty estimates to make an operational decision.

---

## Features

- Automatic download and checksum verification of the M5 dataset
- Memory-conscious preprocessing with Polars
- Store-level Parquet partitioning
- Synthetic lead-time, purchase-cost, and shortage-penalty generation
- Reusable discrete time-series feature engineering
- Lag and rolling-window demand features
- Calendar, event, and selling-price features
- Poisson and negative-binomial regression
- Automatic model-family selection using an overdispersion flag
- Multi-horizon probabilistic forecasts
- Empirical and parametric discrete distributions
- Demand-during-lead-time convolution
- Random sums of non-identically distributed demand variables
- Budget-constrained multi-product newsvendor optimization
- Store-level and category-level WMAPE reporting

---

## Repository Structure

The code expects the following module names:

```text
.
├── process_m5_dataset.py
├── run_m5_pipeline.py
├── probability_utils.py
├── discrete_distribution.py
├── time_series.py
├── multi_product_newsvendor.py
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
│       ├── predictions_<STORE_ID>.csv
│       ├── wmape_store.csv
│       └── wmape_store_cat.csv
└── README.md
```

### Module Overview

#### `process_m5_dataset.py`

Handles data acquisition and preprocessing:

- downloads the M5 archive from Zenodo;
- verifies the archive using an MD5 checksum;
- extracts the raw files;
- reshapes daily demand from wide to long format;
- joins calendar data with weekly selling prices;
- generates synthetic lead-time and newsvendor inputs;
- saves calendar and price tables as Parquet;
- partitions sales data by store.

#### `time_series.py`

Contains the forecasting framework:

- constructs complete time-series panels;
- creates train/test splits;
- generates lag and rolling-window features;
- adds external numerical and categorical features;
- fits candidate historical demand distributions;
- trains grouped Poisson or negative-binomial models;
- generates multi-horizon forecasts;
- converts daily forecasts into lead-time demand distributions;
- computes point-forecast error components.

#### `discrete_distribution.py`

Implements discrete probability distributions and operations:

- creation from empirical observations;
- creation from supported parametric families;
- probability mass functions and cumulative distributions;
- quantiles;
- shifting;
- convolution;
- bin merging and splitting;
- identically distributed random sums;
- non-identically distributed random sums.

The currently supported parametric families are:

- negative binomial;
- binomial;
- Poisson;
- geometric;
- discrete uniform.

#### `probability_utils.py`

Provides utility functions for:

- method-of-moments distribution fitting;
- discrete PMF fit evaluation;
- plotting observed and fitted distributions.

#### `multi_product_newsvendor.py`

Solves a multi-product newsvendor problem under a shared purchasing budget.

For a fixed Lagrange multiplier, each product receives an adjusted critical fractile. A binary search finds a multiplier that satisfies the total budget constraint.

#### `run_m5_pipeline.py`

Runs the complete forecasting and inventory pipeline one store at a time:

- loads processed inputs;
- constructs forecasting features;
- trains grouped GLMs;
- creates lead-time demand distributions;
- solves the inventory optimization problem;
- writes model data, predictions, inventory targets, and WMAPE summaries.

---

## Methodology

### 1. Probabilistic Demand Forecasting

The current forecasting layer uses count-regression models.

For each training group and forecast horizon:

- a **Poisson model** is used when historical variance does not exceed the mean;
- a **negative-binomial model** is used when the series is overdispersed.

The default forecasting horizon is seven days.

The default demand features are:

- lag 1;
- lag 7;
- lag 14;
- rolling mean over 7 days;
- rolling mean over 28 days;
- day-of-week indicators;
- event type;
- selling price.

Models are grouped by:

- store;
- product category;
- overdispersion status.

### 2. Lead-Time Demand

Daily predictive distributions are combined with an empirical store-level lead-time distribution.

When lead time is uncertain, total demand is a random sum:

```text
D_LT = D_1 + D_2 + ... + D_L
```

where `L` is a random lead time and the daily demand variables may have different distributions.

The implementation supports this through `DiscreteDistribution.random_sum`.

### 3. Multi-Product Newsvendor

Each product has:

- a demand-during-lead-time distribution;
- selling price;
- purchase cost;
- shortage penalty.

The optimizer selects inventory quantities while enforcing:

```text
sum(purchase_cost[i] * inventory[i]) <= available_budget
```

When the unconstrained newsvendor solution exceeds the available budget, the solver performs a binary search over the Lagrange multiplier associated with the budget constraint.

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
```

`pyarrow` is required by Pandas for reading and writing Parquet files.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create a virtual environment:

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
pip install numpy pandas polars scipy statsmodels matplotlib requests pyarrow
```

A `requirements.txt` file can also be created with:

```text
numpy
pandas
polars
scipy
statsmodels
matplotlib
requests
pyarrow
```

Then install it using:

```bash
pip install -r requirements.txt
```

---

## How to Run

The pipeline has two stages:

1. Download and preprocess the M5 data.
2. Train the forecasting models and solve the newsvendor problem.

Run all commands from the repository root so that local module imports resolve correctly.

### Step 1: Download and Process the Dataset

```bash
python process_m5_dataset.py --download
```

This command:

- downloads the M5 dataset if necessary;
- verifies the downloaded archive;
- extracts the raw CSV files;
- generates synthetic lead-time data;
- generates synthetic purchasing-cost and shortage-penalty data;
- creates Parquet input files;
- partitions the sales history by store.

Default paths:

```text
Raw data:
data/m5-forecasting-accuracy/

Processed data:
data/m5-forecasting-accuracy/partitioned_data/

Synthetic data:
data/synthetic_data/
```

To process a dataset that has already been downloaded:

```bash
python process_m5_dataset.py
```

Custom paths can be supplied:

```bash
python process_m5_dataset.py \
  --download \
  --m5-dir data/m5-forecasting-accuracy \
  --synthetic-dir data/synthetic_data \
  --partitioned-dir data/m5-forecasting-accuracy/partitioned_data
```

On Windows PowerShell, the same command can be written on one line:

```powershell
python process_m5_dataset.py --download --m5-dir data/m5-forecasting-accuracy --synthetic-dir data/synthetic_data --partitioned-dir data/m5-forecasting-accuracy/partitioned_data
```

### Step 2: Run Forecasting and Inventory Optimization

```bash
python run_m5_pipeline.py
```

The default inventory budget is `15000` per store.

To use a different budget:

```bash
python run_m5_pipeline.py --budget-per-store 25000
```

Custom input and output directories can also be supplied:

```bash
python run_m5_pipeline.py \
  --synthetic-data-dir data/synthetic_data \
  --m5-data-dir data/m5-forecasting-accuracy/partitioned_data \
  --out-dir data/output \
  --budget-per-store 25000
```

The pipeline processes one store partition at a time to limit peak memory usage.

---

## Outputs

For each store, the pipeline writes:

### Model Data

```text
data/output/model_data_<STORE_ID>.csv
```

Contains fitted model metadata and coefficients for each training group and forecast horizon.

### Predictions and Inventory Decisions

```text
data/output/predictions_<STORE_ID>.csv
```

Contains fields derived during forecasting and optimization, including:

- predicted demand;
- actual demand;
- under-forecast error;
- over-forecast error;
- absolute error;
- selling price;
- purchase cost;
- shortage penalty;
- target inventory.

### Forecasting Metrics

```text
data/output/wmape_store.csv
data/output/wmape_store_cat.csv
```

These files summarize:

- WMAPE;
- under-forecast WMAPE;
- over-forecast WMAPE;

at store and store-category levels.

---

## Using the Components Independently

### Create an Empirical Discrete Distribution

```python
import numpy as np

from discrete_distribution import DiscreteDistribution

observations = np.array([0, 1, 1, 2, 2, 2, 3])
distribution = DiscreteDistribution.from_data(observations)

print(distribution.pmf)
print(distribution.quantile(0.95))
```

### Create a Parametric Distribution

```python
from discrete_distribution import DiscreteDistribution

poisson_demand = DiscreteDistribution.from_parametric(
    "poisson",
    params=(4.5, 0),
)
```

### Convolve Independent Demand Distributions

```python
weekly_demand = poisson_demand

for _ in range(6):
    weekly_demand = weekly_demand + poisson_demand
```

### Compute a Random Sum

```python
import numpy as np

from discrete_distribution import DiscreteDistribution

lead_time = DiscreteDistribution(
    pmf=np.array([0.2, 0.5, 0.3]),
    min_value=1,
)

daily_forecasts = [
    DiscreteDistribution.from_parametric("poisson", (3.0, 0)),
    DiscreteDistribution.from_parametric("poisson", (4.0, 0)),
    DiscreteDistribution.from_parametric("poisson", (5.0, 0)),
]

lead_time_demand = lead_time.random_sum(daily_forecasts)
```

### Solve a Multi-Product Newsvendor Problem

```python
import numpy as np

from discrete_distribution import DiscreteDistribution
from multi_product_newsvendor import solve_multi_product_newsvendor

demand_distributions = [
    DiscreteDistribution.from_parametric("poisson", (8.0, 0)),
    DiscreteDistribution.from_parametric("poisson", (12.0, 0)),
]

inventory = solve_multi_product_newsvendor(
    demand_distributions=demand_distributions,
    selling_price=np.array([10.0, 15.0]),
    purchase_cost=np.array([6.0, 9.0]),
    shortage_penalty=np.array([2.0, 3.0]),
    available_budget=150.0,
)

print(inventory)
```

---

## Configuration

Important defaults are defined near the top of `run_m5_pipeline.py`:

```python
DEFAULT_SYNTHETIC_DATA_DIR = Path("data/synthetic_data")
DEFAULT_M5_DATA_DIR = Path(
    "data/m5-forecasting-accuracy/partitioned_data"
)
DEFAULT_OUT_DIR = Path("data/output")
DEFAULT_BUDGET_PER_STORE = 15_000.0
```

Forecasting defaults are passed when constructing `DiscreteTimeSeries`:

```python
ts = DiscreteTimeSeries(
    sales_df,
    ts_id_cols=[
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ],
    period_col="day",
    var_col="demand",
    num_future_targets=7,
    num_test_days=28,
)
```

The default class-level feature settings are:

```python
lags=(1, 7, 14)
rolling_windows=(7, 28)
```

---

## Data Notes

The project uses real M5 data for:

- historical unit sales;
- calendar attributes;
- events;
- selling prices.

The following inputs are currently synthetic:

- store-level lead-time observations;
- product purchase costs;
- shortage penalties.

Synthetic lead times are generated separately for each store. Purchase costs are sampled as a fraction of the latest observed selling price, while shortage penalties are derived from selling prices.

Therefore, the current optimization outputs should be interpreted as a **methodological demonstration**, not as validated production inventory recommendations.

---

## Performance and Memory

The M5 dataset is large after converting the daily columns into a long table.

The preprocessing stage uses Polars for:

- lazy CSV scanning;
- unpivoting;
- joining;
- Parquet serialization.

Sales data is partitioned by store, and the main pipeline processes those store files sequentially. This reduces memory usage relative to training on the complete M5 panel at once.

Model fitting can still be computationally expensive because multiple GLMs are trained across:

- stores;
- categories;
- overdispersion groups;
- forecast horizons.

For development, consider filtering the data to one store or category before running the complete dataset.

---

## Current Limitations

- Inventory decisions are not yet evaluated through a historical replenishment simulation.
- Lead-time and cost inputs are synthetic.
- The forecasting layer currently supports only GLM-based Poisson and negative-binomial models.
- The negative-binomial implementation should be validated carefully for parameter serialization and reconstruction.
- Discrete probability distributions are truncated when an infinite-support parametric distribution is converted into a finite PMF.
- PMF convolution can become expensive for distributions with wide support.
- The newsvendor solver assumes separable product economics with a shared budget constraint.
- The current forecast reporting focuses on point-error metrics; probabilistic calibration metrics are not yet included.
- Automated tests and continuous integration are not yet included.

---

## Next Steps

### 1. Add Simulation-Based Backtesting

Build a historical inventory simulation that evaluates decisions rather than forecasts alone.

The simulation should:

- move through the test period chronologically;
- generate forecasts using only information available at each decision date;
- sample or replay realized lead times;
- place replenishment orders;
- track outstanding inventory in transit;
- update on-hand inventory after demand realization;
- account for lost sales or backorders;
- calculate holding, purchasing, shortage, and wastage costs;
- compare optimized decisions against baseline policies.

Potential baseline policies:

- order-up-to based on mean demand;
- fixed service-level inventory;
- moving-average replenishment;
- unconstrained newsvendor;
- deterministic forecast plus safety stock.

Recommended decision metrics:

- total profit;
- total cost;
- fill rate;
- cycle service level;
- stockout frequency;
- average inventory;
- inventory turnover;
- wastage or excess stock;
- budget utilization.

This will make it possible to answer the most important question:

> Do better probabilistic forecasts produce better inventory decisions?

### 2. Add CatBoost Quantile Regression

Add CatBoost as a nonlinear forecasting alternative.

The implementation could:

- train separate models for selected quantiles;
- use CatBoost's quantile loss;
- handle categorical variables natively;
- support grouped or global models;
- compare CatBoost with the current Poisson and negative-binomial GLMs;
- reconstruct a discrete predictive distribution from forecast quantiles.

Example quantile levels:

```text
0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95
```

Important design considerations:

- prevent quantile crossing;
- interpolate between estimated quantiles;
- define tail behavior outside the fitted quantile range;
- preserve non-negative integer demand;
- evaluate both calibration and downstream inventory performance.

Possible probabilistic evaluation metrics:

- pinball loss;
- weighted interval score;
- empirical coverage;
- interval width;
- PIT or randomized PIT diagnostics;
- CRPS for discrete demand.

### Additional Improvements

- Add unit tests for distribution algebra and newsvendor optimality.
- Add integration tests for a small M5 subset.
- Add a command-line option to process selected stores or categories.
- Store model metadata in JSON or Parquet rather than CSV-encoded dictionaries.
- Add structured logging and remove debugging prints.
- Add probabilistic calibration reports.
- Add experiment configuration files.
- Add reproducible benchmark scripts.
- Add GitHub Actions for testing and linting.
- Add API documentation and type-checking.
- Parallelize independent store or model-group training where memory permits.

---

## Suggested Development Roadmap

```text
Phase 1 — Forecasting foundation
  ✓ Time-series feature engineering
  ✓ Poisson and negative-binomial regression
  ✓ Multi-horizon prediction
  ✓ Discrete predictive distributions

Phase 2 — Decision layer
  ✓ Lead-time demand calculation
  ✓ Multi-product newsvendor
  ✓ Shared budget constraint

Phase 3 — Evaluation
  □ Rolling-origin forecast backtesting
  □ Inventory simulation
  □ Decision-based metrics
  □ Baseline policy comparison

Phase 4 — Model expansion
  □ CatBoost quantile regression
  □ Quantile-to-distribution reconstruction
  □ Calibration analysis
  □ GLM versus CatBoost benchmarking

Phase 5 — Engineering
  □ Tests
  □ Configuration management
  □ CI/CD
  □ Documentation
```

---

## Reproducibility

Synthetic data generators use NumPy's `default_rng` with a fixed default seed of `0`.

The download script verifies the M5 archive using the expected MD5 checksum before extraction. Partial downloads are written to a temporary `.part` file and renamed only after verification.

For fully reproducible experiments, record:

- Python version;
- dependency versions;
- random seed;
- data paths;
- store/category filters;
- forecasting configuration;
- budget;
- lead-time generation parameters.

---

## Disclaimer

This repository is intended for research, experimentation, and education.

The synthetic lead-time and economic inputs are not representative of any particular retailer. Inventory outputs should not be used as production recommendations without validating the data, assumptions, model calibration, and operational constraints.

---

## Acknowledgements

This project uses the M5 Forecasting Accuracy dataset, originally released for the M5 competition.

The dataset download used by the preprocessing script is hosted on Zenodo.
