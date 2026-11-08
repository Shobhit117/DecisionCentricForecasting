# probabilistic-modeling

**Probabilistic demand forecasting → lead-time demand → inventory decisions → simulation**, in one Python library.

Most forecasting projects stop at a point forecast. Inventory decisions need the **full distribution of demand over the replenishment lead time**, combined with product economics and operational constraints. This library connects all of those steps, with Numba-accelerated distribution algebra underneath.

```text
Sales history + calendar / price / promo covariates
                     │
     DiscreteTimeSeries  (leak-free features, future-known covariates)
                     │
     Forecaster  ── GLM (Poisson / Neg-Binomial)  or  CatBoost multi-quantile
                     │
     Daily discrete predictive distributions
                     │
     DDLT: demand during stochastic lead time + review period  (Numba / FFT)
                     │
     Target inventory ── service-level quantile  or  budgeted multi-product newsvendor
                     │
     Multi-period inventory simulation  →  fill rate, service level, stock, profit
```

> **Status:** active development (v0.3, alpha). The public API may still change. The M5 example uses synthetic lead-time and cost data.

📖 **Full documentation (architecture, models, tutorials):** [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md)
🔍 **Codebase & algorithm deep-dive guide (what/where/why for every module & `ddlt.py` internals):** [`docs/CODEBASE_GUIDE.md`](docs/CODEBASE_GUIDE.md)
🎓 **Theory slides (LaTeX/Beamer):** [`docs/slides/prob_forecasting_main.pdf`](docs/slides/prob_forecasting_main.pdf)

---

## Highlights

- **Forecasting**
  - Panel feature engineering with lags, rolling stats, calendar and external features — no train/test leakage, and production mode with **future-known covariates** (planned promotions, holidays, prices).
  - Grouped **Poisson / negative-binomial GLMs** with automatic overdispersion detection.
  - **CatBoost `MultiQuantile`** regression predicting a whole quantile grid per horizon.
  - Quantile grid → proper discrete PMF (monotone rearrangement, interpolation, tail extrapolation).
  - **Train once, predict many times**: self-contained model bundles (`save` / `load_forecaster` / `predict_from_raw`) that replay feature engineering on new raw data.
- **Uncertainty propagation**
  - `DiscreteDistribution` algebra: convolution, i.i.d. and non-identical random sums, quantiles, CDF, CRPS, sampling.
  - Demand during lead time + review period as a mixture of convolutions, with adaptive direct/FFT Numba kernels (up to ~49× faster on large supports).
- **Decisions & evaluation**
  - Service-level quantile targets or **budget-constrained multi-product newsvendor**.
  - **Numba-parallel inventory simulation** with stochastic lead times, lost sales or backorders, Monte Carlo KPI bands, and economic P&L.
  - Metrics: WMAPE, pinball loss, quantile coverage, CRPS, fill rate, cycle service level, turnover, profit.

---

## Installation

Requires **Python ≥ 3.10**.

```bash
git clone <repo-url>
cd probabilistic_modeling
python -m venv .venv
# Linux/macOS: source .venv/bin/activate     Windows: .venv\Scripts\Activate.ps1
pip install -e .              # core library
pip install -e ".[m5,dev]"    # + M5 example scripts and tests
pip install -e ".[fft]"       # optional: rocket-fft for NumPy FFT inside Numba
```

| Extra | Adds | Needed for |
|---|---|---|
| *(core)* | numpy, pandas, scipy, statsmodels, numba, catboost, matplotlib | the library |
| `m5` | polars, pyarrow, requests | downloading/processing M5, Parquet I/O |
| `fft` | rocket-fft | optional faster FFT path |
| `dev` | pytest (+ `m5` deps) | running the tests |

---

## Quick start

```python
from probabilistic_modeling import (
    DiscreteTimeSeries, GLMForecaster, compute_lead_time_demand,
    DiscreteDistribution, simulate_inventory,
)

# 1. Features from a long panel: one row per (store_id, item_id, day) with integer demand
ts = DiscreteTimeSeries(sales_df, ts_id_cols=["store_id", "item_id"], period_col="day",
                        var_col="demand", num_future_targets=7, num_test_days=28)
ts.add_calendar_features(calendar_df, "wday")

# 2. Probabilistic forecast
forecaster = GLMForecaster(train_groups=["store_id"])
forecaster.fit(ts)
forecaster.save("artifacts/glm_bundle")          # reusable later via load_forecaster()

# 3. Demand during lead time (+ 1-day review period) for every test day
lead_time = {"CA_1": DiscreteDistribution.from_parametric("poisson", (3.0, 1))}
ddlt = compute_lead_time_demand(ts, forecaster, lead_time=lead_time, lead_time_keys=["store_id"],
                                cycle_time=1, return_distribution=True, pred_first_day=False)

# 4. Simulate a 95% service-level policy over the test period
result = simulate_inventory(ddlt, lead_time=lead_time, lead_time_keys=["store_id"],
                            policy="quantile", service_level=0.95, n_mc_simulations=100)
print(result.summary_df[["fill_rate", "cycle_service_level", "avg_on_hand"]])
```

More walkthroughs (CatBoost, model reuse, newsvendor targets, policy comparison, using the distribution algebra on its own, adapting to other domains) are in [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md#6-tutorials).

---

## End-to-end M5 example

Run from the repository root.

```bash
# 1. Download (with MD5 check), reshape and partition M5; generate synthetic lead times & economics
python scripts/process_m5_dataset.py --download

# 2. Forecast → DDLT → targets → simulation, store by store
python scripts/run_m5_pipeline.py --model glm
python scripts/run_m5_pipeline.py --model catboost --catboost-iterations 400 --catboost-train-days 365
python scripts/run_m5_pipeline.py --model glm --target-policy quantile --service-level 0.95

# 3. Is the DDLT calibrated, and does the model beat a naive baseline on decisions?
python scripts/run_m5_pipeline.py --model glm --stores CA_1 --target-policy quantile \
    --evaluate-ddlt --baseline trailing
```

<details>
<summary>All <code>run_m5_pipeline.py</code> options</summary>

| Flag | Default | Description |
|---|---|---|
| `--model` | `glm` | `glm` or `catboost` |
| `--target-policy` | `newsvendor` | `newsvendor` (budgeted) or `quantile` |
| `--service-level` | `0.95` | Quantile used with `--target-policy quantile` |
| `--budget-per-store` | `15000` | Purchasing budget per store per decision day |
| `--stores` | all | Subset of stores, e.g. `--stores CA_1 TX_2` |
| `--evaluate-ddlt` | off | Write DDLT calibration reports (coverage, PIT, dispersion ratio, CRPS) |
| `--baseline` | `none` | `trailing` / `seasonal` empirical-quantile baseline through the same DDLT, policy and simulation |
| `--baseline-window` | `56` | Baseline history window in days |
| `--quantiles` | `0.05 … 0.99` | Quantile grid for CatBoost |
| `--catboost-iterations` | `1000` | Max boosting iterations per horizon |
| `--catboost-train-days` | `730` | Training history window (`0` = all) |
| `--synthetic-data-dir` | `data/synthetic_data` | Lead-time & newsvendor inputs |
| `--m5-data-dir` | `data/m5-forecasting-accuracy/partitioned_data` | Partitioned M5 Parquet files |
| `--out-dir` | `data/output` | Output directory |
| `--log-file` | – | Optional log file |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

</details>

**Outputs** (in `data/output/`, CatBoost files carry a `_catboost` suffix):

| File | Content |
|---|---|
| `glm_models_<STORE>/`, `catboost_models_<STORE>/` | Reusable forecaster bundles (`load_forecaster`) |
| `model_data_<STORE>.csv`, `feature_importance_catboost_<STORE>.csv` | GLM coefficients / CatBoost feature importances |
| `predictions[_catboost]_<STORE>.csv` | Horizon means, quantiles and `target_inventory` per item-day |
| `wmape_store*.csv`, `wmape_store_cat*.csv` | Point accuracy by store / category |
| `quantile_metrics_catboost.csv` | Pinball loss, coverage, CRPS per horizon and quantile |
| `simulation_store*.csv`, `simulation_store_cat*.csv` | Inventory simulation KPIs by store / category (one row per model) |
| `ddlt_calibration_{summary,coverage,pit}*.csv` | DDLT calibration report (`--evaluate-ddlt`) |
| `decision_benchmark*.csv`, `predictions_empirical_*_<STORE>.csv` | Model vs. baseline decision KPIs (`--baseline`) |

---

## Repository layout

```text
src/probabilistic_modeling/
├── distributions/   discrete.py · quantile.py · probability_utils.py
├── models/          forecasters.py · catboost_quantile.py
├── inventory/       newsvendor.py · simulation.py
├── ddlt.py          Numba kernels + DDLTCalculator
├── time_series.py   DiscreteTimeSeries + TimeSeriesFeatureSpec
└── logging_utils.py
scripts/             process_m5_dataset.py · run_m5_pipeline.py
tests/               pytest suite + benchmarks vs. legacy implementation
docs/                DOCUMENTATION.md · slides/ (LaTeX presentation)
data/                (git-ignored) raw, processed and output data
```

---

## Development

```bash
python -m pytest -q                       # full test suite (timing benchmarks are skipped)
python -m pytest -m benchmark -s -q       # speed benchmarks with printed timings
```

Numba kernels are JIT-compiled on first use and cached on disk, so the first run is slower.

### Roadmap

- [x] Leak-free feature engineering, production mode with future-known covariates
- [x] Poisson / NB GLMs and CatBoost multi-quantile models
- [x] Quantile → distribution reconstruction, CRPS
- [x] Numba DDLT with adaptive FFT convolution
- [x] Budgeted multi-product newsvendor
- [x] Reusable forecaster bundles
- [x] Multi-period inventory simulation with decision KPIs
- [x] `src/` package layout
- [x] DDLT calibration report + empirical-quantile baseline
- [x] Numba GLM → DDLT path (~20× faster)
- [ ] Rolling-origin backtesting
- [ ] GLM vs CatBoost decision benchmarking
- [ ] Constrained replenishment (budget, capacity, MOQ) inside the simulation
- [ ] Publish to PyPI

---

## Disclaimer

For research, experimentation and education. The synthetic lead-time and economic inputs do not represent any real retailer; validate data, assumptions, calibration and operational constraints before using outputs for real decisions.

## Acknowledgements

Uses the [M5 Forecasting Accuracy](https://zenodo.org/records/12636070) dataset, originally released for the M5 competition.
