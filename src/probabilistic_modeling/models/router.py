"""
Demand-regime routing forecaster (`models/router.py`).

Implements the practitioner workflow of Long et al. (2023, Section 3.1 & Figure 6):
classify series via Syntetos-Boylan-Croston (SBC) demand categories (`smooth`, `erratic`,
`intermittent`, `lumpy`), fit rich global covariate models (`GLMForecaster` or
`CatBoostQuantileForecaster`) strictly on active/predictable regimes, and route
intermittent / ultra-sparse series to a lightweight empirical or stationary baseline
(`EmpiricalQuantileForecaster`).

Why this reduces training time while improving accuracy
-------------------------------------------------------
In retail and e-commerce assortments, `intermittent` series make up 60%–75%+ of all SKUs
(72.6% in M5, 59.9% in e-commerce; Long et al., 2023, Table 1) and account for the largest
share of GBT / GLM training runtime, yet lag-heavy regression models fail to beat simple
in-sample quantiles on sparse counts. Filtering intermittent SKUs out of the primary
forecaster's training panel shrinks the training matrix by 2x–4x+ while routing sparse
series to an `O(1)` lookup baseline (`EmpiricalQuantileForecaster`).
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import time
from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from probabilistic_modeling.distributions.discrete import DiscreteDistribution
from probabilistic_modeling.distributions.quantile import DEFAULT_QUANTILES, validate_quantile_levels
from probabilistic_modeling.models.baseline import EmpiricalQuantileForecaster
from probabilistic_modeling.models.forecasters import (
    GLMForecaster,
    ProbabilisticForecaster,
    compute_forecasting_metrics,
    load_forecaster,
)

if TYPE_CHECKING:
    from probabilistic_modeling.time_series import DiscreteTimeSeries

logger = logging.getLogger(__name__)


def _normalize_regime_codes(regimes: Sequence[str | int]) -> set[int]:
    """Normalize a collection of SBC regime names or integer codes (0..3) into integer codes."""
    from probabilistic_modeling.time_series import DEMAND_REGIME_CODES, DEMAND_REGIME_NAMES

    codes: set[int] = set()
    for r in regimes:
        if isinstance(r, (int, np.integer)):
            code = int(r)
            if code not in DEMAND_REGIME_NAMES:
                raise ValueError(
                    f"Invalid demand_regime code {code}. Valid codes: {sorted(DEMAND_REGIME_NAMES)}."
                )
            codes.add(code)
        else:
            key = str(r).strip().lower()
            if key not in DEMAND_REGIME_CODES:
                raise ValueError(
                    f"Invalid demand_regime name {r!r}. Valid names: {sorted(DEMAND_REGIME_CODES)}."
                )
            codes.add(DEMAND_REGIME_CODES[key])
    return codes


def _subset_time_series(ts: "DiscreteTimeSeries", allowed_series_df: pd.DataFrame) -> "DiscreteTimeSeries":
    """
    Create a lightweight shallow view of ``ts`` restricted to the series in ``allowed_series_df``
    without re-running panel balancing or lag/rolling feature engineering.
    """
    id_cols = list(ts.ts_id_cols)
    keys_sub = allowed_series_df[id_cols].drop_duplicates().copy()

    sub = copy.copy(ts)
    for col in id_cols:
        if col in ts.df.columns:
            keys_sub[col] = keys_sub[col].astype(ts.df[col].dtype)
    sub.df = pd.merge(ts.df, keys_sub, on=id_cols, how="inner")
    sub.num_ts = len(keys_sub)

    if hasattr(ts, "series_metadata") and not ts.series_metadata.empty:
        meta_keys = keys_sub.copy()
        for col in id_cols:
            if col in ts.series_metadata.columns:
                meta_keys[col] = meta_keys[col].astype(ts.series_metadata[col].dtype)
        sub.series_metadata = pd.merge(ts.series_metadata, meta_keys, on=id_cols, how="inner")

    if hasattr(ts, "raw_df") and isinstance(ts.raw_df, pd.DataFrame) and not ts.raw_df.empty:
        raw_keys = keys_sub.copy()
        for col in id_cols:
            if col in ts.raw_df.columns:
                raw_keys[col] = raw_keys[col].astype(ts.raw_df[col].dtype)
        sub.raw_df = pd.merge(ts.raw_df, raw_keys, on=id_cols, how="inner")

    return sub


def _dense_history_to_frame(hist) -> pd.DataFrame:
    """Reconstruct a compact non-zero + boundary long DataFrame from a `DenseHistory` matrix."""
    n_series, n_periods = hist.matrix.shape
    inv_series = [None] * n_series
    for k, idx in hist.series_index.items():
        inv_series[idx] = k

    rows_idx, cols_idx = np.meshgrid(np.arange(n_series), np.arange(n_periods), indexing="ij")
    flat_r = rows_idx.ravel()
    flat_c = cols_idx.ravel()
    flat_v = hist.matrix.ravel()

    data: dict[str, Any] = {}
    if len(hist.ts_id_cols) == 1:
        s_arr = np.asarray(inv_series, dtype=object)
        data[hist.ts_id_cols[0]] = s_arr[flat_r]
    else:
        for col_pos, col_name in enumerate(hist.ts_id_cols):
            col_vals = np.asarray([t[col_pos] for t in inv_series], dtype=object)
            data[col_name] = col_vals[flat_r]

    data[hist.period_col] = hist.periods[flat_c]
    data["demand"] = flat_v
    return pd.DataFrame(data)


class RegimeRouterForecaster(ProbabilisticForecaster):
    """
    Hybrid forecaster that routes series by Syntetos-Boylan-Croston (SBC) demand regime
    following Long et al. (2023, Figure 6).

    Parameters
    ----------
    primary_forecaster : ProbabilisticForecaster, optional
        Model trained strictly on active/predictable series (by default: `Smooth`, `Erratic`,
        and `Lumpy` series). Defaults to ``GLMForecaster(hurdle=True, velocity_tiers="demand_regime")``.
    intermittent_forecaster : ProbabilisticForecaster, optional
        Fast baseline model used for `intermittent` series (defaults to
        ``EmpiricalQuantileForecaster(method="trailing", window=56)``).
    intermittent_regimes : sequence of str or int, default ("intermittent",)
        SBC demand regimes routed to ``intermittent_forecaster`` (accepts names `'smooth'`,
        `'erratic'`, `'intermittent'`, `'lumpy'` or codes `0, 1, 2, 3`).
    intermittent_zero_threshold : float, optional
        Optional post-first-sale zero-rate cutoff in ``[0, 1]`` (e.g. ``0.95`` as in
        Long et al., 2023, Section 5.4). When set, a series in ``intermittent_regimes`` is
        routed to ``intermittent_forecaster`` *only if* its training ``zero_rate >= threshold``;
        moderately intermittent series below the threshold stay with ``primary_forecaster``.
    quantiles : sequence of float, default DEFAULT_QUANTILES
        Quantile grid populated when unifying outputs across parametric and quantile branches.
    populate_quantiles : bool, default True
        If ``True``, ensures ``predict()`` populates unified ``<target>_q<level>`` columns
        on all rows even when ``primary_forecaster`` is a parametric GLM and
        ``intermittent_forecaster`` is a quantile model.
    save_history : bool, default True
        If ``True``, persists the compact sales history required by ``EmpiricalQuantileForecaster``
        inside ``save()`` so ``load_forecaster()`` restores a ready-to-score bundle.
    """

    forecaster_type = "regime_router"

    def __init__(
        self,
        primary_forecaster: ProbabilisticForecaster | None = None,
        intermittent_forecaster: ProbabilisticForecaster | None = None,
        intermittent_regimes: Sequence[str | int] = ("intermittent",),
        intermittent_zero_threshold: float | None = None,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        populate_quantiles: bool = True,
        save_history: bool = True,
    ):
        super().__init__()
        self.quantiles = tuple(float(q) for q in validate_quantile_levels(quantiles))
        self.primary_forecaster: ProbabilisticForecaster = (
            primary_forecaster
            if primary_forecaster is not None
            else GLMForecaster(hurdle=True, velocity_tiers="demand_regime")
        )
        self.intermittent_forecaster: ProbabilisticForecaster = (
            intermittent_forecaster
            if intermittent_forecaster is not None
            else EmpiricalQuantileForecaster(quantiles=self.quantiles, method="trailing", window=56)
        )
        self.intermittent_regime_codes_: set[int] = _normalize_regime_codes(intermittent_regimes)
        self.intermittent_regimes = tuple(intermittent_regimes)
        if intermittent_zero_threshold is not None and not (0.0 <= float(intermittent_zero_threshold) <= 1.0):
            raise ValueError("'intermittent_zero_threshold' must lie in [0, 1].")
        self.intermittent_zero_threshold = (
            float(intermittent_zero_threshold) if intermittent_zero_threshold is not None else None
        )
        self.populate_quantiles = bool(populate_quantiles)
        self.save_history = bool(save_history)

        self.routing_table_: pd.DataFrame | None = None
        self.routing_summary_: dict[str, int] = {}
        self._fitted_branches: set[str] = set()

    def fit(
        self,
        ts: "DiscreteTimeSeries",
        history: pd.DataFrame | None = None,
    ) -> "RegimeRouterForecaster":
        """
        Classify series on post-first-sale training history, split ``ts`` into primary and
        intermittent partitions, and fit each branch strictly on its assigned series.
        """
        from probabilistic_modeling.time_series import DEMAND_REGIME_NAMES, classify_demand_regimes

        t0 = time.perf_counter()
        self._bind_training_context(ts)

        meta = self.series_metadata_
        if meta is None or meta.empty or "demand_regime" not in meta.columns or "zero_rate" not in meta.columns:
            src_df = (
                history
                if history is not None
                else (
                    ts.raw_df
                    if (hasattr(ts, "raw_df") and isinstance(ts.raw_df, pd.DataFrame) and not ts.raw_df.empty)
                    else ts.df
                )
            )
            train_end = None
            if ts.test_start_period is not None:
                train_Slice = src_df[src_df[ts.period_col] < ts.test_start_period]
            else:
                train_Slice = src_df
            sbc = classify_demand_regimes(
                train_Slice, ts_id_cols=ts.ts_id_cols, period_col=ts.period_col, var_col=ts.var_col, train_end_period=train_end
            )
            if meta is None or meta.empty:
                meta = sbc
            else:
                add_cols = [c for c in sbc.columns if c not in meta.columns]
                if add_cols:
                    meta = pd.merge(meta, sbc[list(ts.ts_id_cols) + add_cols], on=list(ts.ts_id_cols), how="left")
            self.series_metadata_ = meta.copy()

        regimes = meta["demand_regime"].fillna(2).to_numpy(dtype=int)
        in_int_regime = np.isin(regimes, list(self.intermittent_regime_codes_))
        if self.intermittent_zero_threshold is not None and "zero_rate" in meta.columns:
            z_rate = meta["zero_rate"].fillna(1.0).to_numpy(dtype=np.float64)
            is_intermittent_route = in_int_regime & (z_rate >= self.intermittent_zero_threshold)
        else:
            is_intermittent_route = in_int_regime

        routing = meta[list(ts.ts_id_cols)].copy()
        routing["demand_regime"] = regimes
        routing["demand_regime_name"] = pd.Series(regimes, index=routing.index).map(DEMAND_REGIME_NAMES)
        routing["model_route"] = np.where(is_intermittent_route, "intermittent", "primary")
        self.routing_table_ = routing.reset_index(drop=True)

        prim_keys = self.routing_table_[self.routing_table_["model_route"] == "primary"][list(ts.ts_id_cols)]
        int_keys = self.routing_table_[self.routing_table_["model_route"] == "intermittent"][list(ts.ts_id_cols)]

        self._fitted_branches = set()
        n_prim_rows = 0
        n_int_rows = 0

        if len(prim_keys) > 0:
            sub_prim = _subset_time_series(ts, prim_keys)
            n_prim_rows = int((sub_prim.df["is_test"] == 0).sum()) if "is_test" in sub_prim.df.columns else len(sub_prim.df)
            logger.info(
                "RegimeRouterForecaster: fitting primary forecaster (%s) on %d/%d series (%d train rows)...",
                type(self.primary_forecaster).__name__,
                len(prim_keys),
                len(self.routing_table_),
                n_prim_rows,
            )
            self.primary_forecaster.fit(sub_prim)
            self._fitted_branches.add("primary")

        if len(int_keys) > 0:
            sub_int = _subset_time_series(ts, int_keys)
            n_int_rows = int((sub_int.df["is_test"] == 0).sum()) if "is_test" in sub_int.df.columns else len(sub_int.df)
            logger.info(
                "RegimeRouterForecaster: fitting intermittent forecaster (%s) on %d/%d series (%d train rows)...",
                type(self.intermittent_forecaster).__name__,
                len(int_keys),
                len(self.routing_table_),
                n_int_rows,
            )
            if isinstance(self.intermittent_forecaster, EmpiricalQuantileForecaster):
                int_hist = None
                if history is not None:
                    hk = int_keys.copy()
                    for c in ts.ts_id_cols:
                        hk[c] = hk[c].astype(history[c].dtype)
                    int_hist = pd.merge(history, hk, on=list(ts.ts_id_cols), how="inner")
                self.intermittent_forecaster.fit(sub_int, history=int_hist)
            else:
                self.intermittent_forecaster.fit(sub_int)
            self._fitted_branches.add("intermittent")

        self.routing_summary_ = {
            "n_series_total": int(len(self.routing_table_)),
            "n_series_primary": int(len(prim_keys)),
            "n_series_intermittent": int(len(int_keys)),
            "n_train_rows_primary": n_prim_rows,
            "n_train_rows_intermittent": n_int_rows,
        }
        logger.info(
            "RegimeRouterForecaster fitted in %.2fs | primary=%d series (%d rows), intermittent=%d series (%d rows)",
            time.perf_counter() - t0,
            len(prim_keys),
            n_prim_rows,
            len(int_keys),
            n_int_rows,
        )
        return self

    def _assign_routes(self, df: pd.DataFrame) -> np.ndarray:
        """Return a 1D string array of `'primary'` or `'intermittent'` for every row in `df`."""
        if self.routing_table_ is None or not self._fitted_branches:
            raise ValueError("RegimeRouterForecaster has not been fitted yet.")
        default_route = "intermittent" if "intermittent" in self._fitted_branches else "primary"
        if len(self._fitted_branches) == 1:
            only_branch = next(iter(self._fitted_branches))
            return np.full(len(df), only_branch, dtype=object)

        rt = self.routing_table_[self.ts_id_cols_ + ["model_route"]].drop_duplicates(subset=self.ts_id_cols_).copy()
        probe = df[self.ts_id_cols_].copy()
        for c in self.ts_id_cols_:
            if isinstance(probe[c].dtype, pd.CategoricalDtype):
                probe[c] = probe[c].astype(str)
            else:
                probe[c] = probe[c].astype(str)
            rt[c] = rt[c].astype(str)
        merged = pd.merge(probe, rt, on=self.ts_id_cols_, how="left")
        return merged["model_route"].fillna(default_route).to_numpy(dtype=object)

    def predict(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
    ) -> pd.DataFrame:
        """
        Predict across all horizons by dispatching each series to its assigned branch
        (`primary_forecaster` vs. `intermittent_forecaster`) and recombining in original row order.
        """
        from probabilistic_modeling.evaluation.forecast_evaluation import populate_model_quantiles

        test_data, target_cols, _, _ = self._resolve_inference_inputs(ts=ts, df=df, period=period)
        if len(test_data) == 0:
            return test_data.copy()

        routes = self._assign_routes(test_data)
        work = test_data.copy()
        work["_router_orig_pos"] = np.arange(len(work), dtype=np.int64)
        work["model_route"] = routes

        parts: list[pd.DataFrame] = []
        prim_mask = routes == "primary"
        int_mask = ~prim_mask

        if np.any(prim_mask):
            prim_pred = self.primary_forecaster.predict(df=work.loc[prim_mask].copy())
            if self.populate_quantiles:
                prim_pred = populate_model_quantiles(prim_pred, target_cols, quantiles=self.quantiles)
            parts.append(prim_pred)

        if np.any(int_mask):
            int_pred = self.intermittent_forecaster.predict(df=work.loc[int_mask].copy())
            if self.populate_quantiles:
                int_pred = populate_model_quantiles(int_pred, target_cols, quantiles=self.quantiles)
            parts.append(int_pred)

        combined = pd.concat(parts, axis=0, ignore_index=True)
        combined.sort_values("_router_orig_pos", inplace=True, ignore_index=True)
        combined.drop(columns=["_router_orig_pos"], inplace=True)

        if all(t in combined.columns for t in target_cols):
            compute_forecasting_metrics(combined, target_cols, [f"{t}_mu" for t in target_cols])
        return combined

    def compute_ddlt(
        self,
        ts: "DiscreteTimeSeries | pd.DataFrame | None" = None,
        lead_time: dict[Any, DiscreteDistribution] | DiscreteDistribution | None = None,
        lead_time_keys: list[str] | None = None,
        cycle_time: int = 1,
        service_levels: pd.DataFrame | None = None,
        return_distribution: bool = False,
        pred_first_day: bool = True,
        df: pd.DataFrame | None = None,
        model: Any = None,
        period: Any = None,
        max_service_level: float | None = None,
    ) -> pd.DataFrame:
        """
        Compute DDLT distributions or target inventories by dispatching each branch to its
        native compiled DDLT kernel and restoring original row order.
        """
        if lead_time is None:
            raise ValueError("'lead_time' distribution must be provided.")
        test_data, _, _, period_col = self._resolve_inference_inputs(ts=ts, df=df, period=period)
        if pred_first_day and len(test_data) > 0 and period_col in test_data.columns:
            first_day = test_data[period_col].min()
            test_data = test_data[test_data[period_col] == first_day].copy()
        if len(test_data) == 0:
            return test_data.copy()

        routes = self._assign_routes(test_data)
        work = test_data.copy()
        work["_router_orig_pos"] = np.arange(len(work), dtype=np.int64)
        work["model_route"] = routes

        parts: list[pd.DataFrame] = []
        prim_mask = routes == "primary"
        int_mask = ~prim_mask

        if np.any(prim_mask):
            parts.append(
                self.primary_forecaster.compute_ddlt(
                    df=work.loc[prim_mask].copy(),
                    lead_time=lead_time,
                    lead_time_keys=lead_time_keys,
                    cycle_time=cycle_time,
                    service_levels=service_levels,
                    return_distribution=return_distribution,
                    pred_first_day=False,
                    max_service_level=max_service_level,
                )
            )
        if np.any(int_mask):
            parts.append(
                self.intermittent_forecaster.compute_ddlt(
                    df=work.loc[int_mask].copy(),
                    lead_time=lead_time,
                    lead_time_keys=lead_time_keys,
                    cycle_time=cycle_time,
                    service_levels=service_levels,
                    return_distribution=return_distribution,
                    pred_first_day=False,
                    max_service_level=max_service_level,
                )
            )

        combined = pd.concat(parts, axis=0, ignore_index=True)
        combined.sort_values("_router_orig_pos", inplace=True, ignore_index=True)
        combined.drop(columns=["_router_orig_pos"], inplace=True)
        return combined

    def save(self, out_dir: Path | str) -> Path:
        """Persist the router, its routing table, and both sub-forecaster bundles to ``out_dir``."""
        if self.routing_table_ is None or not self._fitted_branches:
            raise ValueError("RegimeRouterForecaster has not been fitted yet.")
        out_path = Path(out_dir)
        self._save_base_context(
            out_path,
            extra_meta={
                "intermittent_regimes": [
                    int(r) if isinstance(r, (int, np.integer)) else str(r) for r in self.intermittent_regimes
                ],
                "intermittent_zero_threshold": self.intermittent_zero_threshold,
                "quantiles": list(self.quantiles),
                "populate_quantiles": self.populate_quantiles,
                "save_history": self.save_history,
                "fitted_branches": sorted(self._fitted_branches),
                "routing_summary": self.routing_summary_,
            },
        )
        self.routing_table_.to_csv(out_path / "routing_table.csv", index=False)
        if "primary" in self._fitted_branches:
            self.primary_forecaster.save(out_path / "primary")
        if "intermittent" in self._fitted_branches:
            self.intermittent_forecaster.save(out_path / "intermittent")
            if (
                self.save_history
                and isinstance(self.intermittent_forecaster, EmpiricalQuantileForecaster)
                and self.intermittent_forecaster.history_ is not None
            ):
                hist_df = _dense_history_to_frame(self.intermittent_forecaster.history_)
                hist_df.rename(columns={"demand": self.intermittent_forecaster.var_col_}, inplace=True)
                hist_df.to_csv(out_path / "intermittent_history.csv", index=False)
        return out_path

    @classmethod
    def load(cls, out_dir: Path | str) -> "RegimeRouterForecaster":
        """Restore a fitted :class:`RegimeRouterForecaster` bundle from ``out_dir``."""
        out_path = Path(out_dir)
        meta = json.loads((out_path / "forecaster_meta.json").read_text(encoding="utf-8"))
        fitted_branches = set(meta.get("fitted_branches", []))

        prim_fc = load_forecaster(out_path / "primary") if "primary" in fitted_branches else None
        int_fc = load_forecaster(out_path / "intermittent") if "intermittent" in fitted_branches else None
        if (
            isinstance(int_fc, EmpiricalQuantileForecaster)
            and (out_path / "intermittent_history.csv").exists()
        ):
            hist_df = pd.read_csv(out_path / "intermittent_history.csv")
            int_fc.update_history(hist_df)

        router = cls(
            primary_forecaster=prim_fc,
            intermittent_forecaster=int_fc,
            intermittent_regimes=meta.get("intermittent_regimes", ["intermittent"]),
            intermittent_zero_threshold=meta.get("intermittent_zero_threshold"),
            quantiles=meta.get("quantiles", DEFAULT_QUANTILES),
            populate_quantiles=bool(meta.get("populate_quantiles", True)),
            save_history=bool(meta.get("save_history", True)),
        )
        router._load_base_context(out_path, meta)
        router._fitted_branches = fitted_branches
        router.routing_summary_ = dict(meta.get("routing_summary", {}))
        rt_path = out_path / "routing_table.csv"
        if rt_path.exists():
            router.routing_table_ = pd.read_csv(rt_path)
        return router
