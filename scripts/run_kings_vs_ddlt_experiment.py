r"""
Inventory Replenishment Simulation Benchmark: King's Formula vs. Discrete Convolution (DDLT).

What this script does
---------------------
Compares classical Gaussian safety-stock inventory control (**King's Formula**) against
probabilistic **Demand-During-Lead-Time (DDLT) discrete convolution** inventory control
across fixed target instock quantiles ``alpha in {0.80, 0.85, 0.90, 0.95, 0.99}`` using
the multi-period Numba inventory simulator (:func:`simulate_inventory`).

Policies Compared
-----------------
1. **King's Formula Baselines** (:func:`compute_kings_target_inventory`):
   $$\mu_{\text{DDLT}} = (\mathbb{E}[L] + R)\,\hat{\mu}_D, \quad
     \sigma_{\text{King}} = \sqrt{(\mathbb{E}[L] + R)\,\hat{\sigma}_D^2 + \text{Var}(L)\,\hat{\mu}_D^2}$$
   $$S_{\text{King}}(\alpha) = \text{round\_or\_ceil}\!\big(\max(0,\, \mu_{\text{DDLT}} + \Phi^{-1}(\alpha)\,\sigma_{\text{King}})\big)$$
   - ``kings_trailing_round`` : Trailing 56-day empirical $\hat{\mu}_D, \hat{\sigma}_D$ with nearest-integer rounding.
   - ``kings_trailing_ceil``  : Trailing 56-day empirical $\hat{\mu}_D, \hat{\sigma}_D$ with ceiling rounding.
   - ``kings_seasonal_round`` : Same-weekday 8-week seasonal $\hat{\mu}_D, \hat{\sigma}_D$ with nearest-integer rounding.

2. **Discrete Convolution (DDLT) Policies** (:class:`DDLTCalculator` + :func:`resolve_target_inventory`):
   Propagates daily predictive distributions $D_{t+h}$ over the stochastic lead-time mixture
   of discrete convolutions $\mathbb{P}(D^{L+R}_t = k) = \sum_l \mathbb{P}(L = l)(f_t * \cdots * f_{t+l+R-1})(k)$
   and sets $S_{\text{DDLT}}(\alpha) = F_{D^{L+R}_t}^{-1}(\alpha)$:
   - ``ddlt_glm``                : Standard Poisson / NB2 GLM + discrete convolution.
   - ``ddlt_glm_hurdle``         : Two-part Zero-Hurdle (Logit + ZTP / ZTNB2) GLM + discrete convolution.
   - ``ddlt_glm_hurdle_round``   : Zero-Hurdle GLM + nearest-integer interpolated discrete quantile.
   - ``ddlt_catboost``           : Multi-Quantile CatBoost Regressor + discrete convolution (optional).
   - ``ddlt_empirical_trailing`` : Covariate-free trailing empirical quantiles + discrete convolution.
   - ``ddlt_empirical_seasonal`` : Covariate-free seasonal empirical quantiles + discrete convolution.

Outputs
-------
Saved to ``--output-dir`` (default ``data/output/experiments_kings_vs_ddlt``):
- ``kings_vs_ddlt_by_section.csv``  : Per-(store, section, lead_time_profile, policy, service_level) KPIs.
- ``kings_vs_ddlt_summary.csv``     : Pooled multi-section KPI comparison at each nominal quantile (80%..99%).
- ``kings_vs_ddlt_by_cat.csv``      : Category-level (FOODS, HOBBIES, HOUSEHOLD) KPI breakdown.
- ``kings_vs_ddlt_iso_service.csv`` : Iso-Service (Pareto frontier) comparison quantifying the exact
  inventory & ordering reduction achieved by DDLT while matching or beating King's realized service levels.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

try:  # allow running from repository root without `pip install -e .`
    import probabilistic_modeling  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probabilistic_modeling import (
    CatBoostQuantileForecaster,
    DEFAULT_QUANTILES,
    DiscreteTimeSeries,
    EmpiricalQuantileForecaster,
    GLMForecaster,
    build_m5_sections,
    compute_kings_target_inventory,
    resolve_target_inventory,
    setup_logging,
    simulate_inventory,
)
from scripts.run_m5_experiments import (
    ALL_M5_STORES,
    DEFAULT_LEAD_TIME_PROFILES,
    build_lead_time_map_for_profile,
    build_section_time_series,
    load_enriched_calendar,
)

DEFAULT_TARGET_QUANTILES = (0.80, 0.85, 0.90, 0.95, 0.99)

ALL_SIM_MODELS = (
    "glm",
    "glm_hurdle",
    "catboost",
    "empirical_trailing",
    "empirical_seasonal",
)


def aggregate_simulation_across_sections(
    sec_df: pd.DataFrame,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Pool multi-section inventory simulation KPIs across all ``(store_id, section)`` folds.
    """
    if sec_df.empty:
        return pd.DataFrame()

    key_cols = [
        *(["lead_time_profile"] if "lead_time_profile" in sec_df.columns else []),
        *(extra_group_cols or []),
        "policy",
        "policy_family",
        "target_quantile",
    ]
    key_cols = [c for c in key_cols if c in sec_df.columns]

    rows: list[dict] = []
    for keys, g in sec_df.groupby(key_cols, dropna=False, observed=True):
        keys_tuple = keys if isinstance(keys, tuple) else (keys,)
        kd = dict(zip(key_cols, keys_tuple))
        w = g["n_item_days"].to_numpy(dtype=np.float64)
        w_sum = float(w.sum())
        if w_sum <= 0:
            continue

        tot_demand = int(g["total_demand"].sum())
        tot_sold = int(g["total_sold"].sum())
        tot_lost = int(g["total_lost_sales"].sum())
        tot_ordered = int(g["total_ordered"].sum())
        tot_arrivals = int(g["total_arrivals"].sum())
        init_inv_total = int(g["initial_inventory_total"].sum()) if "initial_inventory_total" in g.columns else 0
        tot_deployed = init_inv_total + tot_ordered

        fill_rate = float(tot_sold / tot_demand) if tot_demand > 0 else 1.0
        stockout_freq = float(np.average(g["stockout_frequency"], weights=w))
        cycle_sl = 1.0 - stockout_freq
        avg_target = float(np.average(g["avg_target_inventory"], weights=w))
        avg_on_hand = float(np.average(g["avg_on_hand"], weights=w))
        avg_closing = float(np.average(g["avg_closing_inv"], weights=w))
        avg_in_transit = float(np.average(g["avg_in_transit"], weights=w))
        turnover = float(tot_sold / avg_closing) if avg_closing > 0.0 else np.nan

        rows.append(
            {
                **kd,
                "n_sections": int(len(g)),
                "n_item_days": int(w_sum),
                "total_demand": tot_demand,
                "total_sold": tot_sold,
                "total_lost_sales": tot_lost,
                "total_ordered": tot_ordered,
                "initial_inventory_total": init_inv_total,
                "total_stock_deployed": tot_deployed,
                "total_arrivals": tot_arrivals,
                "cycle_service_level": round(cycle_sl, 4),
                "fill_rate": round(fill_rate, 4),
                "stockout_frequency": round(stockout_freq, 4),
                "avg_target_inventory": round(avg_target, 2),
                "avg_on_hand": round(avg_on_hand, 2),
                "avg_closing_inv": round(avg_closing, 2),
                "avg_in_transit": round(avg_in_transit, 2),
                "inventory_turnover": round(turnover, 4) if np.isfinite(turnover) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_iso_service_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an Iso-Service (Pareto Efficiency Frontier) comparison table showing:
    1. For each King's Formula benchmark row at target quantile ``q_king in {0.80, 0.85, 0.90, 0.95, 0.99}``,
    2. Finds the DDLT policy row (and piecewise-linear Pareto interpolation along the DDLT curve)
       that achieves **at least the same Cycle Service Level and Fill Rate** as King's Formula,
    3. Quantifies the exact inventory reduction (``target_inv_saving_pct``, ``on_hand_saving_pct``,
       ``closing_inv_saving_pct``, ``stock_deployed_saving_pct``) and service-level gain.
    """
    if summary_df.empty:
        return pd.DataFrame()

    prof_cols = ["lead_time_profile"] if "lead_time_profile" in summary_df.columns else []
    groups = summary_df.groupby(prof_cols, observed=True) if prof_cols else [((), summary_df)]

    rows: list[dict] = []
    for prof_key, gdf in groups:
        prof_dict = dict(zip(prof_cols, prof_key if isinstance(prof_key, tuple) else (prof_key,)))
        king_policies = [p for p in ("kings_trailing_ceil", "kings_trailing_round") if p in gdf["policy"].values]
        ddlt_policies = [
            p
            for p in gdf["policy"].unique()
            if str(p).startswith("ddlt_")
        ]

        for king_pol in king_policies:
            k_sub = gdf[gdf["policy"] == king_pol].sort_values("target_quantile")
            for _, k_row in k_sub.iterrows():
                q_k = float(k_row["target_quantile"])
                k_csl = float(k_row["cycle_service_level"])
                k_fr = float(k_row["fill_rate"])
                k_target = float(k_row["avg_target_inventory"])
                k_oh = float(k_row["avg_on_hand"])
                k_close = float(k_row["avg_closing_inv"])
                k_ord = int(k_row["total_ordered"])
                k_dep = int(k_row["total_stock_deployed"])

                for ddlt_pol in ddlt_policies:
                    d_sub = gdf[gdf["policy"] == ddlt_pol].sort_values("target_quantile")
                    if d_sub.empty:
                        continue

                    # (a) Same nominal quantile row comparison:
                    same_q = d_sub[np.isclose(d_sub["target_quantile"], q_k)]
                    # (b) Lowest-inventory discrete quantile row that achieves CSL >= k_csl and FR >= k_fr:
                    dominating = d_sub[(d_sub["cycle_service_level"] >= k_csl - 1e-4) & (d_sub["fill_rate"] >= k_fr - 1e-4)]
                    if not dominating.empty:
                        match_row = dominating.iloc[0]
                    else:
                        match_row = d_sub.iloc[-1]

                    m_q = float(match_row["target_quantile"])
                    m_csl = float(match_row["cycle_service_level"])
                    m_fr = float(match_row["fill_rate"])
                    m_target = float(match_row["avg_target_inventory"])
                    m_oh = float(match_row["avg_on_hand"])
                    m_close = float(match_row["avg_closing_inv"])
                    m_ord = int(match_row["total_ordered"])
                    m_dep = int(match_row["total_stock_deployed"])

                    # (c) Piecewise-linear Pareto frontier interpolation at exact Fill Rate = k_fr:
                    d_frs = d_sub["fill_rate"].to_numpy(dtype=np.float64)
                    d_targets = d_sub["avg_target_inventory"].to_numpy(dtype=np.float64)
                    d_ohs = d_sub["avg_on_hand"].to_numpy(dtype=np.float64)
                    if k_fr <= d_frs[0]:
                        interp_target = float(d_targets[0])
                        interp_oh = float(d_ohs[0])
                    elif k_fr >= d_frs[-1]:
                        interp_target = float(d_targets[-1])
                        interp_oh = float(d_ohs[-1])
                    else:
                        interp_target = float(np.interp(k_fr, d_frs, d_targets))
                        interp_oh = float(np.interp(k_fr, d_frs, d_ohs))

                    same_target = float(same_q["avg_target_inventory"].iloc[0]) if not same_q.empty else np.nan
                    same_oh = float(same_q["avg_on_hand"].iloc[0]) if not same_q.empty else np.nan
                    same_csl = float(same_q["cycle_service_level"].iloc[0]) if not same_q.empty else np.nan
                    same_fr = float(same_q["fill_rate"].iloc[0]) if not same_q.empty else np.nan

                    rows.append(
                        {
                            **prof_dict,
                            "king_policy": king_pol,
                            "ddlt_policy": ddlt_pol,
                            "king_quantile": q_k,
                            "king_csl": round(k_csl, 4),
                            "king_fill_rate": round(k_fr, 4),
                            "king_avg_target_inv": round(k_target, 2),
                            "king_avg_on_hand": round(k_oh, 2),
                            "king_avg_closing_inv": round(k_close, 2),
                            "king_total_ordered": k_ord,
                            "king_stock_deployed": k_dep,
                            "same_q_ddlt_csl": round(same_csl, 4),
                            "same_q_ddlt_fill_rate": round(same_fr, 4),
                            "same_q_ddlt_avg_target_inv": round(same_target, 2),
                            "same_q_ddlt_avg_on_hand": round(same_oh, 2),
                            "matched_ddlt_quantile": m_q,
                            "matched_ddlt_csl": round(m_csl, 4),
                            "matched_ddlt_fill_rate": round(m_fr, 4),
                            "matched_ddlt_avg_target_inv": round(m_target, 2),
                            "matched_ddlt_avg_on_hand": round(m_oh, 2),
                            "matched_ddlt_avg_closing_inv": round(m_close, 2),
                            "matched_ddlt_total_ordered": m_ord,
                            "matched_ddlt_stock_deployed": m_dep,
                            "matched_target_inv_saving_pct": round(100.0 * (k_target - m_target) / k_target, 2) if k_target > 0 else 0.0,
                            "matched_on_hand_saving_pct": round(100.0 * (k_oh - m_oh) / k_oh, 2) if k_oh > 0 else 0.0,
                            "matched_ordered_saving_pct": round(100.0 * (k_ord - m_ord) / k_ord, 2) if k_ord > 0 else 0.0,
                            "matched_deployed_saving_pct": round(100.0 * (k_dep - m_dep) / k_dep, 2) if k_dep > 0 else 0.0,
                            "pareto_interp_target_inv_at_king_fr": round(interp_target, 2),
                            "pareto_interp_on_hand_at_king_fr": round(interp_oh, 2),
                            "pareto_target_inv_saving_pct": round(100.0 * (k_target - interp_target) / k_target, 2) if k_target > 0 else 0.0,
                            "pareto_on_hand_saving_pct": round(100.0 * (k_oh - interp_oh) / k_oh, 2) if k_oh > 0 else 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare King's safety-stock formula against DDLT discrete convolution across target quantiles."
    )
    parser.add_argument(
        "--stores",
        type=str,
        nargs="+",
        default=["CA_1"],
        help="M5 store IDs to evaluate (e.g. CA_1 TX_1 WI_1 or 'all').",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["glm", "glm_hurdle", "empirical_trailing", "empirical_seasonal"],
        choices=[*ALL_SIM_MODELS, "all"],
        help="DDLT forecasting models to compare against King's formula.",
    )
    parser.add_argument(
        "--service-levels",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_QUANTILES),
        help="Target instock quantiles to simulate (default: 0.80 0.85 0.90 0.95 0.99).",
    )
    parser.add_argument(
        "--lead-time-profiles",
        type=str,
        nargs="+",
        default=["short_concentrated"],
        choices=[*DEFAULT_LEAD_TIME_PROFILES.keys(), "all"],
        help="Lead-time profile(s) to simulate (default: short_concentrated).",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=365,
        help="Training window length in days per section (default: 365).",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=28,
        help="Simulation test window length in days per section (default: 28).",
    )
    parser.add_argument(
        "--step-days",
        type=int,
        default=365,
        help="Stride in days between sections (default: 365).",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=2,
        help="Maximum number of temporal sections per store (default: 2).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=14,
        help="Daily forecast horizon H (must cover max(L) + cycle_time; default: 14).",
    )
    parser.add_argument(
        "--cycle-time",
        type=int,
        default=1,
        help="Replenishment review period R in days (default: 1).",
    )
    parser.add_argument(
        "--kings-window",
        type=int,
        default=56,
        help="Trailing window in days for King's formula mean/std estimation (default: 56).",
    )
    parser.add_argument(
        "--initial-inventory-mode",
        type=str,
        default="common_cycle_stock",
        choices=["common_cycle_stock", "target_day_0", "zero"],
        help=(
            "How day-0 on-hand inventory is initialized across policies: "
            "'common_cycle_stock' starts every policy at the same SKU-level expected DDLT cycle stock "
            "round((E[L]+R)*mu_D) on day 0 so safety-stock differences are ordered on day 0 and reflected "
            "in total_ordered; 'target_day_0' starts each policy at its own day-0 target."
        ),
    )
    parser.add_argument(
        "--sample-skus",
        type=int,
        default=None,
        help="Optional number of SKUs to sample per store for rapid testing.",
    )
    parser.add_argument(
        "--downsample-zeros-frac",
        type=float,
        default=0.2,
        help="Fraction of all-zero training rows retained during GLM/CatBoost fitting (default: 0.2).",
    )
    parser.add_argument(
        "--catboost-iterations",
        type=int,
        default=500,
        help="Boosting iterations for CatBoostQuantileForecaster (default: 500).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of CPU threads for parallel model fitting (default: -1).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for stochastic lead-time draws in InventorySimulator (default: 42).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/m5-forecasting-accuracy/partitioned_data",
        help="Directory containing partitioned M5 Parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/output/experiments_kings_vs_ddlt",
        help="Directory to save simulation benchmark CSV tables and logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Save per-(store, section) checkpoints and resume from already-completed sections if interrupted.",
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
        log_file=output_dir / "kings_vs_ddlt.log",
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    stores_to_run = (
        list(ALL_M5_STORES)
        if "all" in [s.lower() for s in args.stores]
        else list(dict.fromkeys(args.stores))
    )
    models_to_run = (
        list(ALL_SIM_MODELS)
        if "all" in args.models
        else list(dict.fromkeys(args.models))
    )
    lt_profiles_to_run = (
        list(DEFAULT_LEAD_TIME_PROFILES.keys())
        if "all" in args.lead_time_profiles
        else list(dict.fromkeys(args.lead_time_profiles))
    )
    service_levels = tuple(sorted(float(q) for q in args.service_levels))
    quantiles = tuple(sorted(set(DEFAULT_QUANTILES) | set(service_levels)))
    data_dir = Path(args.data_dir)

    logger.info("=====================================================================")
    logger.info("Inventory Simulation Benchmark: King's Formula vs. DDLT Convolution")
    logger.info("  Stores                 : %s", stores_to_run)
    logger.info("  DDLT Models            : %s", models_to_run)
    logger.info("  Target Quantiles       : %s", list(service_levels))
    logger.info("  Lead-Time Profiles     : %s", lt_profiles_to_run)
    logger.info(
        "  Sectioning             : train_days=%d, test_days=%d, step_days=%d, max_sections=%s",
        args.train_days,
        args.test_days,
        args.step_days,
        args.max_sections,
    )
    logger.info(
        "  Replenishment          : cycle_time=%d, kings_window=%d, init_mode=%s",
        args.cycle_time,
        args.kings_window,
        args.initial_inventory_mode,
    )
    logger.info("  Output directory       : %s (resume=%s)", output_dir, args.resume)
    logger.info("=====================================================================")

    ckpt_dir = output_dir / "checkpoints"
    if args.resume:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    calendar_df = load_enriched_calendar(data_dir)
    prices_df = pd.read_parquet(data_dir / "price.parquet")
    prices_df["sell_price"] = np.log1p(prices_df["sell_price"])
    lead_time_maps = {
        prof: build_lead_time_map_for_profile(prof) for prof in lt_profiles_to_run
    }

    sec_overall_frames: list[pd.DataFrame] = []
    sec_cat_frames: list[pd.DataFrame] = []

    glm_phase_models = [
        m for m in models_to_run if m in ("glm", "glm_hurdle", "empirical_trailing", "empirical_seasonal")
    ]
    cb_phase_models = [m for m in models_to_run if m == "catboost"]

    for store_id in stores_to_run:
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

            ckpt_prefix = ckpt_dir / f"{store_id}_{sec.name}"
            if args.resume and ckpt_prefix.with_name(f"{ckpt_prefix.name}_done.flag").exists():
                logger.info("Resuming: loading completed checkpoint for [%s | %s]", store_id, sec.name)
                sec_overall_frames.append(
                    pd.read_csv(ckpt_prefix.with_name(f"{ckpt_prefix.name}_overall.csv"))
                )
                cat_ckpt = ckpt_prefix.with_name(f"{ckpt_prefix.name}_cat.csv")
                if cat_ckpt.exists():
                    sec_cat_frames.append(pd.read_csv(cat_ckpt))
                continue

            sec_overall_list: list[pd.DataFrame] = []
            sec_cat_list: list[pd.DataFrame] = []

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

            eval_period = list(range(sec.test_start_day, sec.test_end_day + 1))

            # Build GLM/Baseline panel first
            ts_glm = build_section_time_series(
                sales_slice=sales_slice,
                calendar_slice=cal_slice,
                prices_slice=price_slice,
                section=sec,
                horizon=args.horizon,
                for_catboost=False,
                downsample_zeros_frac=args.downsample_zeros_frac,
            )
            base_test_df = ts_glm.get_inference_data(period=eval_period)[
                [*ts_glm.ts_id_cols, "day", "target_t_plus_0"]
            ].copy()

            def _resolve_initial_inventory(
                base_df: pd.DataFrame,
                lt_map: dict,
            ) -> str | np.ndarray:
                if args.initial_inventory_mode == "target_day_0":
                    return "target_day_0"
                if args.initial_inventory_mode == "zero":
                    return "zero"
                # Common initial inventory across all policies on day 0: expected cycle stock round((E[L]+R)*mu_D)
                day0_df = (
                    base_df[base_df["day"] == sec.test_start_day]
                    .sort_values(["store_id", "item_id"])
                    .reset_index(drop=True)
                )
                k_day0 = compute_kings_target_inventory(
                    day0_df,
                    lead_time=lt_map,
                    lead_time_keys=["store_id", "cat_id"],
                    cycle_time=args.cycle_time,
                    service_level=0.50,  # z_0.50 = 0.0 -> pure expected DDLT cycle stock
                    history=sales_slice,
                    ts_id_cols=["store_id", "item_id"],
                    period_col="day",
                    var_col="demand",
                    window=args.kings_window,
                    method="trailing",
                    rounding="round",
                )
                return k_day0["target_inventory"].to_numpy(dtype=np.int64)

            def _record_simulation(
                decided_df: pd.DataFrame,
                lt_prof_name: str,
                lt_map: dict,
                policy_name: str,
                policy_family: str,
                q_lv: float,
                init_inv: str | np.ndarray,
            ) -> None:
                sim_res = simulate_inventory(
                    decided_df,
                    lead_time=lt_map,
                    lead_time_keys=["store_id", "cat_id"],
                    cycle_time=args.cycle_time,
                    policy="precomputed",
                    initial_inventory=init_inv,
                    group_cols=["cat_id"],
                    random_seed=args.random_seed,
                )
                day0_init_sum = int(
                    sim_res.trajectory_df.loc[
                        sim_res.trajectory_df["day"] == sec.test_start_day, "opening_inv"
                    ].sum()
                )
                s_df = sim_res.summary_df.copy()
                s_df["initial_inventory_total"] = day0_init_sum
                s_df["total_stock_deployed"] = day0_init_sum + int(s_df["total_ordered"].iloc[0])
                s_df.insert(0, "target_quantile", float(q_lv))
                s_df.insert(0, "policy_family", policy_family)
                s_df.insert(0, "policy", policy_name)
                if len(lead_time_maps) > 1:
                    s_df.insert(0, "lead_time_profile", lt_prof_name)
                s_df.insert(0, "section", sec.name)
                s_df.insert(0, "section_id", sec.section_id)
                s_df.insert(0, "store_id", store_id)
                sec_overall_frames.append(s_df)
                sec_overall_list.append(s_df)

                if sim_res.group_summary_df is not None:
                    c_df = sim_res.group_summary_df.copy()
                    cat_init = (
                        sim_res.trajectory_df[sim_res.trajectory_df["day"] == sec.test_start_day]
                        .groupby("cat_id", observed=True)["opening_inv"]
                        .sum()
                        .to_dict()
                    )
                    c_df["initial_inventory_total"] = c_df["cat_id"].map(cat_init).fillna(0).astype(int)
                    c_df["total_stock_deployed"] = c_df["initial_inventory_total"] + c_df["total_ordered"]
                    c_df.insert(0, "target_quantile", float(q_lv))
                    c_df.insert(0, "policy_family", policy_family)
                    c_df.insert(0, "policy", policy_name)
                    if len(lead_time_maps) > 1:
                        c_df.insert(0, "lead_time_profile", lt_prof_name)
                    c_df.insert(0, "section", sec.name)
                    c_df.insert(0, "section_id", sec.section_id)
                    c_df.insert(0, "store_id", store_id)
                    sec_cat_frames.append(c_df)
                    sec_cat_list.append(c_df)

            # Precompute trailing and seasonal daily demand moments once per section for King's Formula:
            from probabilistic_modeling._panel import DenseHistory
            dense_hist = DenseHistory.from_frame(sales_slice, ["store_id", "item_id"], "day", "demand")
            r_idx, c_idx = dense_hist.locate(base_test_df)
            hist_mat = dense_hist.matrix
            for k_m in ("trailing", "seasonal"):
                if k_m == "seasonal":
                    n_wks = max(2, int(args.kings_window) // 7)
                    offs = -7 * np.arange(1, n_wks + 1)
                else:
                    offs = np.arange(-max(2, int(args.kings_window)), 0)
                idx_m = c_idx[:, None] + offs[None, :]
                val_mask = (idx_m >= 0) & (idx_m < hist_mat.shape[1]) & (r_idx[:, None] >= 0) & (c_idx[:, None] >= 0)
                vals_m = np.where(
                    val_mask,
                    hist_mat[np.clip(r_idx, 0, None)[:, None], np.clip(idx_m, 0, hist_mat.shape[1] - 1)],
                    np.nan,
                )
                n_v = np.sum(np.isfinite(vals_m), axis=1)
                mu_d_m = np.nan_to_num(np.where(n_v > 0, np.nanmean(vals_m, axis=1), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
                std_d_m = np.nan_to_num(np.where(n_v > 1, np.nanstd(vals_m, axis=1, ddof=1), 0.0), nan=0.0, posinf=0.0, neginf=0.0)
                base_test_df[f"_kings_mu_d_{k_m}_{int(args.kings_window)}"] = mu_d_m
                base_test_df[f"_kings_std_d_{k_m}_{int(args.kings_window)}"] = std_d_m

            # 1. Evaluate King's Formula Baselines across all lead-time profiles and target quantiles
            init_inv_by_profile: dict[str, str | np.ndarray] = {}
            for lt_prof_name, lt_map in lead_time_maps.items():
                init_inv = _resolve_initial_inventory(base_test_df, lt_map)
                init_inv_by_profile[lt_prof_name] = init_inv

                for king_name, k_method, k_round in (
                    ("kings_trailing_round", "trailing", "round"),
                    ("kings_trailing_ceil", "trailing", "ceil"),
                    ("kings_seasonal_round", "seasonal", "round"),
                ):
                    for q_lv in service_levels:
                        k_df = compute_kings_target_inventory(
                            base_test_df,
                            lead_time=lt_map,
                            lead_time_keys=["store_id", "cat_id"],
                            cycle_time=args.cycle_time,
                            service_level=q_lv,
                            history=sales_slice,
                            ts_id_cols=["store_id", "item_id"],
                            period_col="day",
                            var_col="demand",
                            window=args.kings_window,
                            method=k_method,
                            rounding=k_round,
                        )
                        _record_simulation(
                            k_df,
                            lt_prof_name,
                            lt_map,
                            policy_name=king_name,
                            policy_family="kings_formula",
                            q_lv=q_lv,
                            init_inv=init_inv,
                        )

            # 2. Evaluate DDLT (Discrete Convolution) Forecasters
            for phase_models, is_cb in ((glm_phase_models, False), (cb_phase_models, True)):
                if not phase_models:
                    continue
                if is_cb:
                    del ts_glm
                    ts_glm = None
                    gc.collect()
                    ts_cb = build_section_time_series(
                        sales_slice=sales_slice,
                        calendar_slice=cal_slice,
                        prices_slice=price_slice,
                        section=sec,
                        horizon=args.horizon,
                        for_catboost=True,
                        downsample_zeros_frac=args.downsample_zeros_frac,
                    )
                else:
                    ts_cb = None

                for model_name in phase_models:
                    m_t0 = time.perf_counter()
                    if model_name == "glm":
                        assert ts_glm is not None
                        forecaster = GLMForecaster(
                            train_groups=["store_id", "cat_id"],
                            hurdle=False,
                            velocity_tiers=True,
                            n_jobs=args.n_jobs,
                        )
                        forecaster.fit(ts_glm)
                        active_ts = ts_glm
                    elif model_name == "glm_hurdle":
                        assert ts_glm is not None
                        forecaster = GLMForecaster(
                            train_groups=["store_id", "cat_id"],
                            hurdle=True,
                            velocity_tiers="demand_regime",
                            n_jobs=args.n_jobs,
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
                            window=args.kings_window,
                            horizon=args.horizon,
                        )
                        forecaster.fit(ts_glm, history=sales_slice)
                        active_ts = ts_glm
                    elif model_name == "empirical_seasonal":
                        assert ts_glm is not None
                        forecaster = EmpiricalQuantileForecaster(
                            quantiles=quantiles,
                            method="seasonal",
                            n_weeks=max(2, args.kings_window // 7),
                            horizon=args.horizon,
                        )
                        forecaster.fit(ts_glm, history=sales_slice)
                        active_ts = ts_glm
                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    pred_eval_df = forecaster.predict(active_ts, period=eval_period)
                    for lt_prof_name, lt_map in lead_time_maps.items():
                        init_inv = init_inv_by_profile[lt_prof_name]
                        ddlt_dist_df = forecaster.compute_ddlt(
                            active_ts,
                            df=pred_eval_df,
                            lead_time=lt_map,
                            lead_time_keys=["store_id", "cat_id"],
                            cycle_time=args.cycle_time,
                            return_distribution=True,
                            pred_first_day=False,
                        )

                        # Standard discrete quantile F^{-1}(q) = min{k : F(k) >= q}
                        for q_lv in service_levels:
                            decided_ceil = resolve_target_inventory(
                                ddlt_dist_df,
                                policy="quantile",
                                service_level=q_lv,
                                quantile_rounding="ceil",
                            ).drop(columns=["demand_distribution"])
                            _record_simulation(
                                decided_ceil,
                                lt_prof_name,
                                lt_map,
                                policy_name=f"ddlt_{model_name}",
                                policy_family="ddlt_convolution",
                                q_lv=q_lv,
                                init_inv=init_inv,
                            )

                            # Also record nearest-integer interpolated discrete quantile for GLM/Hurdle/CatBoost
                            if model_name in ("glm", "glm_hurdle", "catboost"):
                                decided_round = resolve_target_inventory(
                                    ddlt_dist_df,
                                    policy="quantile",
                                    service_level=q_lv,
                                    quantile_rounding="round",
                                ).drop(columns=["demand_distribution"])
                                _record_simulation(
                                    decided_round,
                                    lt_prof_name,
                                    lt_map,
                                    policy_name=f"ddlt_{model_name}_round",
                                    policy_family="ddlt_convolution",
                                    q_lv=q_lv,
                                    init_inv=init_inv,
                                )

                        del ddlt_dist_df

                    logger.info(
                        "Finished DDLT simulations for '%s' on %s / %s in %.2fs",
                        model_name,
                        store_id,
                        sec.name,
                        time.perf_counter() - m_t0,
                    )
                    del forecaster
                    gc.collect()

            if ts_glm is not None:
                del ts_glm
            gc.collect()
            if args.resume:
                pd.concat(sec_overall_list, ignore_index=True).to_csv(
                    ckpt_prefix.with_name(f"{ckpt_prefix.name}_overall.csv"), index=False
                )
                if sec_cat_list:
                    pd.concat(sec_cat_list, ignore_index=True).to_csv(
                        ckpt_prefix.with_name(f"{ckpt_prefix.name}_cat.csv"), index=False
                    )
                ckpt_prefix.with_name(f"{ckpt_prefix.name}_done.flag").write_text("ok", encoding="utf-8")
            logger.info("Completed section %s / %s in %.2fs", store_id, sec.name, time.perf_counter() - sec_t0)

    by_sec_df = pd.concat(sec_overall_frames, ignore_index=True)
    by_sec_df.to_csv(output_dir / "kings_vs_ddlt_by_section.csv", index=False)

    summary_df = aggregate_simulation_across_sections(by_sec_df)
    summary_df.to_csv(output_dir / "kings_vs_ddlt_summary.csv", index=False)

    if sec_cat_frames:
        cat_sec_df = pd.concat(sec_cat_frames, ignore_index=True)
        by_cat_df = aggregate_simulation_across_sections(cat_sec_df, extra_group_cols=["cat_id"])
        by_cat_df.to_csv(output_dir / "kings_vs_ddlt_by_cat.csv", index=False)

    iso_df = build_iso_service_comparison(summary_df)
    iso_df.to_csv(output_dir / "kings_vs_ddlt_iso_service.csv", index=False)

    display_cols = [
        *(["lead_time_profile"] if "lead_time_profile" in summary_df.columns else []),
        "target_quantile",
        "policy",
        "cycle_service_level",
        "fill_rate",
        "avg_target_inventory",
        "avg_on_hand",
        "avg_closing_inv",
        "total_ordered",
        "total_stock_deployed",
        "total_lost_sales",
    ]
    logger.info(
        "\n=== Pooled Simulation Summary: King's Formula vs. DDLT Convolution ===\n%s",
        summary_df[[c for c in display_cols if c in summary_df.columns]]
        .sort_values([*(["lead_time_profile"] if "lead_time_profile" in summary_df.columns else []), "target_quantile", "policy"])
        .to_string(index=False),
    )

    if not iso_df.empty:
        iso_cols = [
            *(["lead_time_profile"] if "lead_time_profile" in iso_df.columns else []),
            "king_policy",
            "ddlt_policy",
            "king_quantile",
            "king_csl",
            "king_fill_rate",
            "king_avg_target_inv",
            "king_avg_on_hand",
            "king_total_ordered",
            "matched_ddlt_quantile",
            "matched_ddlt_csl",
            "matched_ddlt_fill_rate",
            "matched_ddlt_avg_target_inv",
            "matched_ddlt_avg_on_hand",
            "matched_ddlt_total_ordered",
            "matched_target_inv_saving_pct",
            "matched_on_hand_saving_pct",
            "matched_ordered_saving_pct",
            "pareto_target_inv_saving_pct",
            "pareto_on_hand_saving_pct",
        ]
        focus_iso = iso_df[iso_df["ddlt_policy"].isin(["ddlt_glm_hurdle", "ddlt_glm_hurdle_round", "ddlt_glm", "ddlt_catboost"])]
        logger.info(
            "\n=== Iso-Service & Pareto Frontier Efficiency Summary (DDLT vs. King's Formula) ===\n%s",
            focus_iso[[c for c in iso_cols if c in focus_iso.columns]].to_string(index=False),
        )

    logger.info("All simulation tables saved to %s", output_dir)


if __name__ == "__main__":
    main()
