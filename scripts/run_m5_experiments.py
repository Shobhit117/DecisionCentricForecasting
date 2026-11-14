"""
Multi-Section M5 Forecasting Benchmark: Multi-Quantile WMAPE & PIT Evaluation.

What this script does
---------------------
Trains and evaluates all 5 probabilistic forecasters implemented in ``probabilistic_modeling``
across multiple temporal sections of the M5 dataset (by default: 1-year training windows
``train_days=365`` forecasting the next ``test_days=14`` days):

1. ``glm``                : Standard Poisson / Negative Binomial (NB2) GLM routed by series overdispersion
2. ``glm_hurdle``         : Two-part Zero-Hurdle (Logit + Zero-Truncated Poisson / NB2) GLM
3. ``catboost``           : Multi-Quantile CatBoost Regressor (``MultiQuantile:alpha=...``)
4. ``empirical_trailing`` : Covariate-free trailing-window empirical quantile baseline (56-day window)
5. ``empirical_seasonal`` : Covariate-free same-weekday seasonal empirical quantile baseline (8 weeks)

Evaluation Framework
--------------------
(i) **WMAPE across Quantiles** (:func:`evaluate_quantile_wmape`):
    Computes ``wmape``, ``under_wmape``, ``over_wmape``, ``pinball_loss``, ``pinball_wmape``, and
    empirical ``coverage`` across every quantile ``q in quantiles`` (plus ``"mean"``) at:
    - Each individual horizon ``target_t_plus_0 .. target_t_plus_{H-1}``,
    - Pooled across all horizons (``target="all_horizons"``),
    - Summed across the 14-day horizon per decision row (``target="horizon_sum"``),
    - Overall and broken down by ``cat_id``.

(ii) **Probability Integral Transform (PIT) Distributions** (:func:`evaluate_forecast_pit` &
     :func:`evaluate_ddlt_calibration`):
    Evaluates the non-randomized discrete PIT of Czado, Gneiting & Held (2009) both at the
    daily horizon level and after convolving daily forecasts over stochastic replenishment
    lead times (DDLT), producing:
    - PIT decile histograms (``density = 1.0`` under a calibrated forecast),
    - PIT uniformity summary metrics (``pit_mean``, ``pit_std``, ``pit_mae``, ``pit_rmse``,
      ``pit_ks``, ``pit_tail_lower``, ``pit_tail_upper``, ``crps``, ``dispersion_ratio``),
    - Raw coverage vs. discrete-corrected ``pit_coverage`` and ``gap_pp`` at target service levels,
    - DDLT quantile WMAPE and pinball loss across target service levels.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

try:  # allow running from the repository root without `pip install -e .`
    import probabilistic_modeling  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from probabilistic_modeling import (
    CatBoostQuantileForecaster,
    DEFAULT_QUANTILES,
    DiscreteDistribution,
    DiscreteTimeSeries,
    EmpiricalQuantileForecaster,
    GLMForecaster,
    M5Section,
    build_m5_sections,
    compute_pit_uniformity_metrics,
    evaluate_ddlt_calibration,
    evaluate_forecast_pit,
    evaluate_quantile_wmape,
    populate_model_quantiles,
    setup_logging,
)
from probabilistic_modeling.evaluation.ddlt_calibration import DEFAULT_CALIBRATION_LEVELS

ALL_MODELS = (
    "glm",
    "glm_hurdle",
    "catboost",
    "empirical_trailing",
    "empirical_seasonal",
)


def build_section_time_series(
    sales_slice: pd.DataFrame,
    calendar_slice: pd.DataFrame,
    prices_slice: pd.DataFrame,
    section: M5Section,
    horizon: int,
    for_catboost: bool = False,
    downsample_zeros_frac: float | None = 0.2,
) -> DiscreteTimeSeries:
    """
    Construct the panel :class:`DiscreteTimeSeries` for a single ``(store, section)`` window.

    Uses the exact same temporal lags, rolling windows, calendar features, and external
    price/event covariates as ``scripts/run_m5_pipeline.py``.
    """
    ts_id_cols = [
        c
        for c in ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        if c in sales_slice.columns
    ]
    unique_days = np.sort(sales_slice["day"].unique())
    num_test_days = int((unique_days >= section.test_start_day).sum())

    ts = DiscreteTimeSeries(
        df=sales_slice,
        ts_id_cols=ts_id_cols,
        period_col="day",
        var_col="demand",
        num_future_targets=horizon,
        num_test_days=num_test_days,
        include_std_dev=for_catboost,
        use_one_hot_encoding=not for_catboost,
        downsample_zeros_frac=downsample_zeros_frac,
        fit_historical_distributions=True,
    )
    ts.add_calendar_features(calendar_slice, "wday")
    ts.add_feature(prices_slice, "sell_price", join_on=["store_id", "item_id"])

    event_col = (
        "event_type"
        if "event_type" in calendar_slice.columns
        else ("event_type_1" if "event_type_1" in calendar_slice.columns else None)
    )
    if event_col is not None:
        ts.add_feature(
            calendar_slice.fillna({event_col: "no_event"}),
            event_col,
            join_on=[],
            is_cat_feature=True,
            default_value="no_event",
        )
    if for_catboost and "month" in calendar_slice.columns:
        ts.add_feature(calendar_slice, "month", join_on=[])
    return ts


def build_default_lead_time_map() -> dict[tuple[str, str], DiscreteDistribution]:
    """Build per-(store_id, cat_id) stochastic lead-time distributions for DDLT evaluation."""
    stores = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    categories = ["FOODS", "HOBBIES", "HOUSEHOLD"]
    return {
        (s, c): DiscreteDistribution(np.array([0.2, 0.5, 0.3]), 1, 1)
        for s in stores
        for c in categories
    }


def aggregate_quantile_wmape_across_sections(
    wmape_df: pd.DataFrame,
    group_filter: str = "overall",
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate per-section quantile WMAPE tables across all ``(store, section)`` runs.

    Computes both:
    - **Pooled (volume-weighted) metrics**: exact ratio ``100 * sum(abs_diff) / sum(actual)``
      across all sections and stores, and
    - **Section-level distribution metrics**: ``wmape_section_mean`` and ``wmape_section_std``
      across sections.
    """
    sub = wmape_df[wmape_df["group"] == group_filter].copy()
    if extra_group_cols:
        sub = wmape_df[wmape_df["group"] != "overall"].copy()
    if sub.empty:
        return pd.DataFrame()

    key_cols = ["model", *(extra_group_cols or []), "target", "quantile", "quantile_level"]
    key_cols = [c for c in key_cols if c in sub.columns]

    rows: list[dict] = []
    for keys, g in sub.groupby(key_cols, dropna=False, observed=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        kd = dict(zip(key_cols, keys_tuple))
        n_sections = int(len(g))
        n_obs = int(g["n_obs"].sum())
        actual = float(g["actual"].sum())
        pred = float(g["pred"].sum())
        under_diff = float(g["under_diff"].sum())
        over_diff = float(g["over_diff"].sum())
        abs_diff = under_diff + over_diff

        q_lv = kd.get("quantile_level", np.nan)
        if actual > 0.0:
            wmape = 100.0 * abs_diff / actual
            under_wmape = 100.0 * under_diff / actual
            over_wmape = 100.0 * over_diff / actual
        else:
            wmape = np.nan
            under_wmape = np.nan
            over_wmape = np.nan

        if pd.notna(q_lv) and np.isfinite(float(q_lv)):
            tau = float(q_lv)
            pinball_sum = tau * under_diff + (1.0 - tau) * over_diff
            pinball_loss = pinball_sum / n_obs if n_obs > 0 else np.nan
            pinball_wmape = 100.0 * 2.0 * pinball_sum / actual if actual > 0.0 else np.nan
        else:
            pinball_loss = abs_diff / (2.0 * n_obs) if n_obs > 0 else np.nan
            pinball_wmape = wmape

        cov = float(np.average(g["coverage"], weights=g["n_obs"])) if n_obs > 0 else np.nan
        rows.append(
            {
                **kd,
                "n_sections": n_sections,
                "n_obs": n_obs,
                "actual": actual,
                "pred": pred,
                "under_diff": under_diff,
                "over_diff": over_diff,
                "abs_diff": abs_diff,
                "wmape": wmape,
                "wmape_section_mean": float(g["wmape"].mean()),
                "wmape_section_std": float(g["wmape"].std(ddof=0)),
                "under_wmape": under_wmape,
                "over_wmape": over_wmape,
                "pinball_loss": pinball_loss,
                "pinball_wmape": pinball_wmape,
                "coverage": cov,
            }
        )
    return pd.DataFrame(rows)


def aggregate_pit_across_sections(
    summary_sec_df: pd.DataFrame,
    cov_sec_df: pd.DataFrame,
    pit_sec_df: pd.DataFrame,
    obs_col: str = "n_obs",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pool PIT summary tables, coverage tables, and PIT histograms across all ``(store, section)``
    folds weighted by observation count ``obs_col``.
    """
    has_target = "target" in summary_sec_df.columns
    group_keys = ["model", "group", *(["target"] if has_target else [])]

    # 1. Aggregate PIT histograms first so pooled uniformity metrics use pooled bin masses
    pit_rows: list[dict] = []
    pooled_pit_map: dict[tuple, np.ndarray] = {}
    for keys, g in pit_sec_df.groupby(group_keys, dropna=False, observed=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        kd = dict(zip(group_keys, keys_tuple))
        bins_sorted = sorted(g["bin"].unique())
        n_bins = len(bins_sorted)
        mass_vec = np.zeros(n_bins, dtype=np.float64)
        for idx, b in enumerate(bins_sorted):
            gb = g[g["bin"] == b]
            w = gb[obs_col].to_numpy(dtype=np.float64) if obs_col in gb.columns else np.ones(len(gb))
            w_sum = float(w.sum())
            m_val = float(np.average(gb["mass"], weights=w)) if w_sum > 0 else float(gb["mass"].mean())
            mass_vec[idx] = m_val
            pit_rows.append(
                {
                    **kd,
                    "bin": int(b),
                    "lower": float(gb["lower"].iloc[0]),
                    "upper": float(gb["upper"].iloc[0]),
                    "mass": m_val,
                    "density": m_val * n_bins,
                    "density_section_std": float(gb["density"].std(ddof=0)),
                }
            )
        pooled_pit_map[keys_tuple] = mass_vec

    # 2. Aggregate PIT summary metrics
    summ_rows: list[dict] = []
    for keys, g in summary_sec_df.groupby(group_keys, dropna=False, observed=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        kd = dict(zip(group_keys, keys_tuple))
        w = g[obs_col].to_numpy(dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue
        mp = float(np.average(g["mean_pred"], weights=w))
        mr = float(np.average(g["mean_realized"], weights=w))
        crps = float(np.average(g["crps"], weights=w))
        disp_valid = g["dispersion_ratio"].notna()
        disp = (
            float(np.average(g.loc[disp_valid, "dispersion_ratio"], weights=w[disp_valid]))
            if disp_valid.any()
            else np.nan
        )
        u_mean = float(np.average(g["pit_mean"], weights=w))
        u_sq = float(np.average(g["pit_std"] ** 2 + g["pit_mean"] ** 2, weights=w))
        uni = compute_pit_uniformity_metrics(pooled_pit_map[keys_tuple], u_mean=u_mean, u_sq=u_sq)
        summ_rows.append(
            {
                **kd,
                "n_sections": int(len(g)),
                obs_col: int(w_sum),
                "mean_pred": mp,
                "mean_realized": mr,
                "bias_pct": 100.0 * (mp - mr) / mr if mr > 0 else np.nan,
                "crps": crps,
                "dispersion_ratio": disp,
                **uni,
            }
        )

    # 3. Aggregate coverage & DDLT quantile WMAPE table
    cov_keys = [*group_keys, "level"]
    cov_rows: list[dict] = []
    for keys, g in cov_sec_df.groupby(cov_keys, dropna=False, observed=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        kd = dict(zip(cov_keys, keys_tuple))
        w = g[obs_col].to_numpy(dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue
        tau = float(kd["level"])
        cv = float(np.average(g["coverage"], weights=w))
        pcv = float(np.average(g["pit_coverage"], weights=w))
        row_dict: dict = {
            **kd,
            "n_sections": int(len(g)),
            obs_col: int(w_sum),
            "coverage": cv,
            "pit_coverage": pcv,
            "gap_pp": 100.0 * (pcv - tau),
        }
        if "actual" in g.columns and "under_diff" in g.columns:
            actual = float(g["actual"].sum())
            pred = float(g["pred"].sum())
            u_sum = float(g["under_diff"].sum())
            o_sum = float(g["over_diff"].sum())
            a_sum = u_sum + o_sum
            pinball_sum = tau * u_sum + (1.0 - tau) * o_sum
            row_dict.update(
                {
                    "actual": actual,
                    "pred": pred,
                    "under_diff": u_sum,
                    "over_diff": o_sum,
                    "abs_diff": a_sum,
                    "wmape": 100.0 * a_sum / actual if actual > 0 else np.nan,
                    "under_wmape": 100.0 * u_sum / actual if actual > 0 else np.nan,
                    "over_wmape": 100.0 * o_sum / actual if actual > 0 else np.nan,
                    "pinball_loss": pinball_sum / w_sum if w_sum > 0 else np.nan,
                    "pinball_wmape": 100.0 * 2.0 * pinball_sum / actual if actual > 0 else np.nan,
                }
            )
        cov_rows.append(row_dict)

    return pd.DataFrame(summ_rows), pd.DataFrame(cov_rows), pd.DataFrame(pit_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-section M5 forecasting experiments (Quantile WMAPE & PIT evaluation)."
    )
    parser.add_argument(
        "--stores",
        type=str,
        nargs="+",
        default=["CA_1"],
        help="M5 store IDs to evaluate (e.g. CA_1 CA_2 TX_1 WI_1).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(ALL_MODELS),
        choices=[*ALL_MODELS, "all"],
        help="Models to train and evaluate (default: all 5 models).",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=365,
        help="Training window length in days for each M5 section (default: 365, ~1 year).",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=14,
        help="Forecast evaluation window length in days for each section (default: 14).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=365,
        help="Stride in days between consecutive M5 sections (default: 365).",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=None,
        help="Optional cap on the number of sections per store.",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=None,
        help="Optional starting day for the first section (defaults to min day in store parquet).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Number of daily forecast horizons target_t_plus_0 .. target_t_plus_{H-1} (default: 14).",
    )
    parser.add_argument(
        "--pred-first-day",
        action="store_true",
        help="Evaluate only the first decision day of each test section (non-overlapping 14-day ahead trajectory).",
    )
    parser.add_argument(
        "--sample-skus",
        type=int,
        default=None,
        help="Optional number of SKUs to sample per store for rapid smoke testing.",
    )
    parser.add_argument(
        "--downsample-zeros-frac",
        type=float,
        default=0.2,
        help="Fraction of zero-demand training rows retained during GLM/CatBoost fitting (default: 0.2).",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=500,
        help="Number of boosting trees for CatBoostQuantileForecaster (default: 500).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel CPU workers for GLM/CatBoost training (default: -1, all cores).",
    )
    parser.add_argument(
        "--trailing-window",
        type=int,
        default=56,
        help="Trailing window in days for empirical_trailing baseline (default: 56).",
    )
    parser.add_argument(
        "--seasonal-weeks",
        type=int,
        default=8,
        help="Number of historical same-weekday weeks for empirical_seasonal baseline (default: 8).",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUANTILES),
        help="Quantile levels to evaluate for WMAPE and quantile forecasters.",
    )
    parser.add_argument(
        "--calibration-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_CALIBRATION_LEVELS),
        help="Target service levels for PIT coverage evaluation (default: 0.5 0.8 0.9 0.95 0.99).",
    )
    parser.add_argument(
        "--n-pit-bins",
        type=int,
        default=10,
        help="Number of histogram bins for PIT distributions (default: 10 deciles).",
    )
    parser.add_argument(
        "--skip-ddlt",
        action="store_true",
        help="Skip multi-day lead-time demand (DDLT) convolution and DDLT PIT evaluation.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/m5-forecasting-accuracy/partitioned_data",
        help="Directory containing processed M5 parquet files (default: data/m5-forecasting-accuracy/partitioned_data).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/experiments",
        help="Directory where experiment CSV tables and logs will be saved.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(
        level=args.log_level,
        log_file=output_dir / "m5_experiments.log",
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    models_to_run = list(ALL_MODELS) if "all" in args.models else list(dict.fromkeys(args.models))
    quantiles = tuple(sorted(float(q) for q in args.quantiles))
    calib_levels = tuple(sorted(float(lv) for lv in args.calibration_levels))
    data_dir = Path(args.data_dir)

    logger.info("=====================================================================")
    logger.info("M5 Probabilistic Forecasting Benchmark")
    logger.info("  Stores             : %s", args.stores)
    logger.info("  Models             : %s", models_to_run)
    logger.info(
        "  Sectioning         : train_days=%d, test_days=%d, step_days=%d, max_sections=%s",
        args.train_days,
        args.test_days,
        args.step_days,
        args.max_sections,
    )
    logger.info("  Forecast horizon   : %d days (pred_first_day=%s)", args.horizon, args.pred_first_day)
    logger.info("  Quantiles          : %s", list(quantiles))
    logger.info("  Calibration levels : %s (n_pit_bins=%d)", list(calib_levels), args.n_pit_bins)
    logger.info("  Output directory   : %s", output_dir)
    logger.info("=====================================================================")

    calendar_df = pd.read_parquet(data_dir / "calendar.parquet")
    prices_df = pd.read_parquet(data_dir / "price.parquet")
    prices_df["sell_price"] = np.log1p(prices_df["sell_price"])
    lead_time_map = build_default_lead_time_map()

    manifest_rows: list[dict] = []
    wmape_sec_frames: list[pd.DataFrame] = []
    fpit_summ_frames: list[pd.DataFrame] = []
    fpit_cov_frames: list[pd.DataFrame] = []
    fpit_hist_frames: list[pd.DataFrame] = []
    ddlt_summ_frames: list[pd.DataFrame] = []
    ddlt_cov_frames: list[pd.DataFrame] = []
    ddlt_hist_frames: list[pd.DataFrame] = []

    need_glm_ts = any(
        m in models_to_run for m in ("glm", "glm_hurdle", "empirical_trailing", "empirical_seasonal")
    )
    need_cb_ts = "catboost" in models_to_run

    for store_id in args.stores:
        sales_path = data_dir / f"sales_data_{store_id}.parquet"
        if not sales_path.exists():
            raise FileNotFoundError(
                f"Missing {sales_path}. Run `python scripts/process_m5_dataset.py` first."
            )
        sales_df = pd.read_parquet(sales_path)
        if args.sample_skus is not None and args.sample_skus > 0:
            keep_items = sorted(sales_df["item_id"].unique())[: int(args.sample_skus)]
            sales_df = sales_df[sales_df["item_id"].isin(keep_items)].reset_index(drop=True)
            logger.info("Store %s: sampled %d SKUs (%d rows)", store_id, len(keep_items), len(sales_df))

        # Pre-filter price table to this store once (reduces 60M rows -> ~6M rows)
        store_prices_df = prices_df[prices_df["store_id"] == store_id].reset_index(drop=True)

        min_day = int(sales_df["day"].min())
        max_day = int(sales_df["day"].max())
        sections = build_m5_sections(
            min_day=min_day,
            max_day=max_day,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            max_sections=args.max_sections,
            start_day=args.start_day,
        )
        logger.info(
            "Store %s (days %d..%d): built %d sections of %d train days + %d test days",
            store_id,
            min_day,
            max_day,
            len(sections),
            args.train_days,
            args.test_days,
        )

        for sec in sections:
            sec_t0 = time.perf_counter()
            logger.info(
                "--- [%s | %s] train=[%d..%d] (%dd), test=[%d..%d] (%dd) ---",
                store_id,
                sec.name,
                sec.train_start_day,
                sec.train_end_day,
                sec.train_days,
                sec.test_start_day,
                sec.test_end_day,
                sec.test_days,
            )
            manifest_rows.append(
                {
                    "store_id": store_id,
                    "section_id": sec.section_id,
                    "section": sec.name,
                    "train_start_day": sec.train_start_day,
                    "train_end_day": sec.train_end_day,
                    "train_days": sec.train_days,
                    "test_start_day": sec.test_start_day,
                    "test_end_day": sec.test_end_day,
                    "test_days": sec.test_days,
                }
            )

            # Include horizon-1 lookahead days after test_end_day so all test decision days
            # have complete realized targets target_t_plus_0..target_t_plus_{H-1} and DDLT windows.
            slice_end_day = min(max_day, sec.test_end_day + args.horizon - 1)
            day_mask = (sales_df["day"] >= sec.train_start_day) & (sales_df["day"] <= slice_end_day)
            sales_slice = sales_df.loc[day_mask].reset_index(drop=True)
            cal_slice = calendar_df[
                (calendar_df["day"] >= sec.train_start_day) & (calendar_df["day"] <= slice_end_day)
            ].reset_index(drop=True)
            price_slice = store_prices_df[
                (store_prices_df["day"] >= sec.train_start_day)
                & (store_prices_df["day"] <= slice_end_day)
            ].reset_index(drop=True)

            ts_glm: DiscreteTimeSeries | None = None
            ts_cb: DiscreteTimeSeries | None = None
            if need_glm_ts:
                fe_t0 = time.perf_counter()
                ts_glm = build_section_time_series(
                    sales_slice=sales_slice,
                    calendar_slice=cal_slice,
                    prices_slice=price_slice,
                    section=sec,
                    horizon=args.horizon,
                    for_catboost=False,
                    downsample_zeros_frac=args.downsample_zeros_frac,
                )
                logger.info(
                    "Built GLM/Baseline panel (%d rows, %d features) in %.2fs",
                    len(ts_glm.df),
                    len(ts_glm.feature_cols),
                    time.perf_counter() - fe_t0,
                )
            if need_cb_ts:
                fe_t0 = time.perf_counter()
                ts_cb = build_section_time_series(
                    sales_slice=sales_slice,
                    calendar_slice=cal_slice,
                    prices_slice=price_slice,
                    section=sec,
                    horizon=args.horizon,
                    for_catboost=True,
                    downsample_zeros_frac=args.downsample_zeros_frac,
                )
                logger.info(
                    "Built CatBoost panel (%d rows, %d features) in %.2fs",
                    len(ts_cb.df),
                    len(ts_cb.feature_cols),
                    time.perf_counter() - fe_t0,
                )

            eval_period = (
                [sec.test_start_day]
                if args.pred_first_day
                else list(range(sec.test_start_day, sec.test_end_day + 1))
            )

            for model_name in models_to_run:
                m_t0 = time.perf_counter()
                logger.info("Running model '%s' on %s / %s...", model_name, store_id, sec.name)

                if model_name == "glm":
                    assert ts_glm is not None
                    forecaster = GLMForecaster(
                        train_groups=["store_id", "cat_id"], hurdle=False, n_jobs=args.n_jobs
                    )
                    forecaster.fit(ts_glm)
                    active_ts = ts_glm
                elif model_name == "glm_hurdle":
                    assert ts_glm is not None
                    forecaster = GLMForecaster(
                        train_groups=["store_id", "cat_id"], hurdle=True, n_jobs=args.n_jobs
                    )
                    forecaster.fit(ts_glm)
                    active_ts = ts_glm
                elif model_name == "catboost":
                    assert ts_cb is not None
                    forecaster = CatBoostQuantileForecaster(
                        quantiles=quantiles,
                        catboost_params={
                            "iterations": args.catboost_iterations,
                            "learning_rate": 0.08,
                            "depth": 6,
                            "thread_count": args.n_jobs,
                            "verbose": False,
                        },
                    )
                    forecaster.fit(ts_cb)
                    active_ts = ts_cb
                elif model_name == "empirical_trailing":
                    assert ts_glm is not None
                    forecaster = EmpiricalQuantileForecaster(
                        quantiles=quantiles,
                        method="trailing",
                        window=args.trailing_window,
                        horizon=args.horizon,
                    )
                    forecaster.fit(ts_glm, history=sales_slice)
                    active_ts = ts_glm
                elif model_name == "empirical_seasonal":
                    assert ts_glm is not None
                    forecaster = EmpiricalQuantileForecaster(
                        quantiles=quantiles,
                        method="seasonal",
                        n_weeks=args.seasonal_weeks,
                        horizon=args.horizon,
                    )
                    forecaster.fit(ts_glm, history=sales_slice)
                    active_ts = ts_glm
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                target_cols = list(active_ts.target_cols)
                pred_df = forecaster.predict(active_ts, period=eval_period)
                populate_model_quantiles(pred_df, target_cols, quantiles=quantiles, inplace=True)

                # (i) Evaluate WMAPE across quantiles
                wmape_sec = evaluate_quantile_wmape(
                    pred_df=pred_df,
                    target_cols=target_cols,
                    quantiles=quantiles,
                    group_cols=["cat_id"],
                    include_mean=True,
                    include_horizon_sum=True,
                )
                wmape_sec.insert(0, "model", model_name)
                wmape_sec.insert(0, "section", sec.name)
                wmape_sec.insert(0, "section_id", sec.section_id)
                wmape_sec.insert(0, "store_id", store_id)
                wmape_sec_frames.append(wmape_sec)

                # (ii-a) Evaluate horizon-level PIT distributions
                fpit_res = evaluate_forecast_pit(
                    pred_df=pred_df,
                    target_cols=target_cols,
                    levels=calib_levels,
                    quantiles=quantiles,
                    group_cols=["cat_id"],
                    include_by_horizon=True,
                    n_pit_bins=args.n_pit_bins,
                )
                for df_piece, collector in (
                    (fpit_res.summary_df, fpit_summ_frames),
                    (fpit_res.coverage_df, fpit_cov_frames),
                    (fpit_res.pit_df, fpit_hist_frames),
                ):
                    tagged = df_piece.copy()
                    tagged.insert(0, "model", model_name)
                    tagged.insert(0, "section", sec.name)
                    tagged.insert(0, "section_id", sec.section_id)
                    tagged.insert(0, "store_id", store_id)
                    collector.append(tagged)

                # (ii-b) Evaluate convolved DDLT PIT distributions & DDLT quantile WMAPE
                if not args.skip_ddlt:
                    ddlt_dist_df = forecaster.compute_ddlt(
                        active_ts,
                        lead_time=lead_time_map,
                        lead_time_keys=["store_id", "cat_id"],
                        cycle_time=1,
                        return_distribution=True,
                        pred_first_day=args.pred_first_day,
                        period=eval_period,
                    )
                    ddlt_cal = evaluate_ddlt_calibration(
                        ddlt_df=ddlt_dist_df,
                        history=sales_slice,
                        lead_time=lead_time_map,
                        ts_id_cols=active_ts.ts_id_cols,
                        lead_time_keys=["store_id", "cat_id"],
                        cycle_time=1,
                        period_col=active_ts.period_col,
                        var_col=active_ts.var_col,
                        levels=calib_levels,
                        group_cols=["cat_id"],
                        n_pit_bins=args.n_pit_bins,
                    )
                    for df_piece, collector in (
                        (ddlt_cal.summary_df, ddlt_summ_frames),
                        (ddlt_cal.coverage_df, ddlt_cov_frames),
                        (ddlt_cal.pit_df, ddlt_hist_frames),
                    ):
                        tagged = df_piece.copy()
                        tagged.insert(0, "model", model_name)
                        tagged.insert(0, "section", sec.name)
                        tagged.insert(0, "section_id", sec.section_id)
                        tagged.insert(0, "store_id", store_id)
                        collector.append(tagged)

                q50_row = wmape_sec[
                    (wmape_sec["group"] == "overall")
                    & (wmape_sec["target"] == "all_horizons")
                    & (wmape_sec["quantile"] == "q0.5")
                ]
                pit_row = fpit_res.summary_df[
                    (fpit_res.summary_df["group"] == "overall")
                    & (fpit_res.summary_df["target"] == "all_horizons")
                ]
                q50_wmape = float(q50_row["wmape"].iloc[0]) if not q50_row.empty else np.nan
                pit_mae = float(pit_row["pit_mae"].iloc[0]) if not pit_row.empty else np.nan
                logger.info(
                    "Finished '%s' on %s / %s in %.2fs | all_horizons q0.5 WMAPE=%.2f%% | PIT MAE=%.3f",
                    model_name,
                    store_id,
                    sec.name,
                    time.perf_counter() - m_t0,
                    q50_wmape,
                    pit_mae,
                )

            logger.info("Completed section %s / %s in %.2fs", store_id, sec.name, time.perf_counter() - sec_t0)

    # -----------------------------------------------------------------------
    # Save Manifest & Aggregated Experiment Tables
    # -----------------------------------------------------------------------
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(output_dir / "sections_manifest.csv", index=False)

    wmape_by_sec_df = pd.concat(wmape_sec_frames, ignore_index=True)
    wmape_by_sec_df.to_csv(output_dir / "quantile_wmape_by_section.csv", index=False)

    wmape_summary_df = aggregate_quantile_wmape_across_sections(wmape_by_sec_df, group_filter="overall")
    wmape_summary_df.to_csv(output_dir / "quantile_wmape_summary.csv", index=False)

    wmape_by_cat_df = aggregate_quantile_wmape_across_sections(
        wmape_by_sec_df, group_filter="cat", extra_group_cols=["cat_id"]
    )
    wmape_by_cat_df.to_csv(output_dir / "quantile_wmape_by_cat.csv", index=False)

    fpit_by_sec_df = pd.concat(fpit_summ_frames, ignore_index=True)
    fpit_cov_sec_df = pd.concat(fpit_cov_frames, ignore_index=True)
    fpit_hist_sec_df = pd.concat(fpit_hist_frames, ignore_index=True)
    fpit_by_sec_df.to_csv(output_dir / "forecast_pit_by_section.csv", index=False)

    fpit_summary_df, fpit_coverage_df, fpit_histogram_df = aggregate_pit_across_sections(
        fpit_by_sec_df, fpit_cov_sec_df, fpit_hist_sec_df, obs_col="n_obs"
    )
    fpit_summary_df.to_csv(output_dir / "forecast_pit_summary.csv", index=False)
    fpit_coverage_df.to_csv(output_dir / "forecast_pit_coverage.csv", index=False)
    fpit_histogram_df.to_csv(output_dir / "forecast_pit_histogram.csv", index=False)

    if ddlt_summ_frames:
        ddlt_by_sec_df = pd.concat(ddlt_summ_frames, ignore_index=True)
        ddlt_cov_sec_df = pd.concat(ddlt_cov_frames, ignore_index=True)
        ddlt_hist_sec_df = pd.concat(ddlt_hist_frames, ignore_index=True)
        ddlt_by_sec_df.to_csv(output_dir / "ddlt_pit_by_section.csv", index=False)

        ddlt_summary_df, ddlt_coverage_df, ddlt_histogram_df = aggregate_pit_across_sections(
            ddlt_by_sec_df, ddlt_cov_sec_df, ddlt_hist_sec_df, obs_col="n_rows"
        )
        ddlt_summary_df.to_csv(output_dir / "ddlt_pit_summary.csv", index=False)
        ddlt_coverage_df.to_csv(output_dir / "ddlt_pit_coverage.csv", index=False)
        ddlt_histogram_df.to_csv(output_dir / "ddlt_pit_histogram.csv", index=False)

    # Log concise comparison tables
    overall_wmape = wmape_summary_df[wmape_summary_df["target"] == "all_horizons"]
    if not overall_wmape.empty:
        pivot_wmape = overall_wmape.pivot(index="model", columns="quantile", values="wmape")
        logger.info("\n=== Pooled Daily Quantile WMAPE (all_horizons, %%) ===\n%s", pivot_wmape.round(2).to_string())

    hsum_wmape = wmape_summary_df[wmape_summary_df["target"] == "horizon_sum"]
    if not hsum_wmape.empty:
        pivot_hsum = hsum_wmape.pivot(index="model", columns="quantile", values="wmape")
        logger.info("\n=== 14-Day Horizon-Sum Quantile WMAPE (horizon_sum, %%) ===\n%s", pivot_hsum.round(2).to_string())

    overall_fpit = fpit_summary_df[
        (fpit_summary_df["group"] == "overall") & (fpit_summary_df["target"] == "all_horizons")
    ]
    if not overall_fpit.empty:
        cols = [
            "model",
            "n_sections",
            "n_obs",
            "bias_pct",
            "crps",
            "dispersion_ratio",
            "pit_mean",
            "pit_std",
            "pit_mae",
            "pit_ks",
            "pit_tail_lower",
            "pit_tail_upper",
        ]
        logger.info(
            "\n=== Daily Forecast PIT Summary (all_horizons) ===\n%s",
            overall_fpit[[c for c in cols if c in overall_fpit.columns]].round(4).to_string(index=False),
        )

    if ddlt_summ_frames:
        overall_ddlt = ddlt_summary_df[ddlt_summary_df["group"] == "overall"]
        if not overall_ddlt.empty:
            cols = [
                "model",
                "n_sections",
                "n_rows",
                "bias_pct",
                "crps",
                "dispersion_ratio",
                "pit_mean",
                "pit_std",
                "pit_mae",
                "pit_ks",
                "pit_tail_lower",
                "pit_tail_upper",
            ]
            logger.info(
                "\n=== Convolved DDLT PIT Summary (overall) ===\n%s",
                overall_ddlt[[c for c in cols if c in overall_ddlt.columns]].round(4).to_string(index=False),
            )

    logger.info("All experiment tables saved to %s", output_dir)


if __name__ == "__main__":
    main()
