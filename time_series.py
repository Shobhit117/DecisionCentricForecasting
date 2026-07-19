import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from discrete_distribution import compute_best_fit_distribution, DiscreteDistribution
import statsmodels.formula.api as smf
from typing import Any
import logging
import json
logging.basicConfig(level=logging.INFO)

def train_glm(df: pd.DataFrame, feature_cols: list[str], target_col: str, model_type: str = 'nbinom') -> tuple[dict, str]:
    """
    Trains a Generalized Linear Model (GLM) on the provided DataFrame.
    """
    train_df = df[~df[target_col].isna()]
    if 'is_test' in train_df.columns:
        train_df = train_df[train_df['is_test'] == 0]
    formula = f'{target_col} ~ {" + ".join(feature_cols)}'
    if model_type == 'nbinom': 
        model = smf.negativebinomial(formula=formula, data=train_df)
    elif model_type == 'poisson':
        model = smf.poisson(formula=formula, data=train_df)
    else:
        raise ValueError(f"Unsupported distribution: '{model_type}'. Supported options are 'nbinom', 'poisson'.")
    result = model.fit(maxiter=100)
    prod_model = {"model_type": model_type
                , "horizon": target_col
                , "coefficients": result.params.to_dict()}
    return prod_model, result.summary()

def predict_glm(model_data: dict[str, Any], feature_dict: dict[str, Any] | pd.Series) -> tuple[float, float | None]:
    """
    Computes the expected value (mu) and dispersion parameter (alpha) 
    for a given feature set using trained GLM coefficients.
    """
    coefficients = model_data['coefficients']
    z = coefficients.get('Intercept', 0.0)
    for feature_name, coef_value in coefficients.items():
        if feature_name != 'Intercept':
            z += feature_dict.get(feature_name, 0.0) * coef_value
    mu = np.exp(z)
    alpha = model_data.get('alpha', np.nan)
    return mu, alpha

def load_glm_model_data(file_name : str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    for col in df.columns:
        if col.startswith('target'):
            df[col] = df[col].apply(json.loads)
    return df

class DiscreteTimeSeries:
    def __init__(self, df : pd.DataFrame, ts_id_cols : list[str], period_col: str, var_col : str, to_datetime : bool = False, default_value : int = 0, lags : tuple[int] = (1, 7, 14), rolling_windows : tuple[int] = (7, 28), num_future_targets : int = 7, num_test_days : int = 1, include_std_dev : bool = False, use_best_fit_feature : bool = False, use_one_hot_encoding : bool = True):
        self.ts_id_cols = ts_id_cols
        self.var_col = var_col
        self.period_col = period_col
        self.df = df[ts_id_cols + [period_col, var_col]].copy()
        for col in ts_id_cols:
            self.df[col] = self.df[col].astype('category')
        if to_datetime:
            self.df[period_col] = pd.to_datetime(self.df[period_col])
        self.start_period = self.df[period_col].min()
        self.end_period = self.df[period_col].max()
        self.periods = sorted(self.df[period_col].unique())
        self.num_periods = len(self.periods)
        self.df[period_col] = self.df[period_col].astype('category')
        self.feature_cols = []
        self.target_cols = []
        self.categorical_feature_cols = []
        self.test_start_period = None
        self.use_one_hot_encoding = use_one_hot_encoding
        self.logger = logging.getLogger(__name__)
        
        # Compute all ts_id x period combinations:
        unique_ts_id = self.df[ts_id_cols].drop_duplicates()
        self.num_ts = len(unique_ts_id)
        unique_ts_id_period = pd.merge(unique_ts_id, pd.DataFrame({period_col: self.periods}), how='cross')
        self.df = pd.merge(unique_ts_id_period, self.df, on=ts_id_cols + [period_col], how='left').fillna({var_col: default_value})

        self._add_test_flag(num_test_days)
        grouped = self.df.groupby(self.ts_id_cols)
        self._add_overdispersion_flag(grouped)
        self._fit_distributions(grouped, use_best_fit_feature, use_one_hot_encoding)
        self._engineer_features(lags, rolling_windows, num_future_targets, include_std_dev)
    
    def _add_test_flag(self, num_test_days : int):
        self.test_start_period = self.periods[-num_test_days]
        self.df['is_test'] = np.where(self.df[self.period_col] >= self.test_start_period, 1, 0)

    def _engineer_features(self, lags : tuple, rolling_windows : tuple, num_future_targets : int, include_std_dev : bool, apply_log_transformation : bool = True):
        feature_df = self.df.copy().sort_values(by = self.ts_id_cols + [self.period_col])
        grouped_df = feature_df.groupby(self.ts_id_cols)
        # Create Lag Features:
        for lag in lags:
            col_name = f'lag_{lag}'
            if apply_log_transformation:
                feature_df[col_name] = np.log1p(grouped_df[self.var_col].shift(lag))
            else:
                feature_df[col_name] = grouped_df[self.var_col].shift(lag)
            self.feature_cols.append(col_name)
        
        # Create Rolling Window Features:
        for window in rolling_windows:
            # Rolling Mean:
            col_name = f'rolling_mean_{window}'
            if apply_log_transformation:
                feature_df[col_name] = np.log1p(grouped_df[self.var_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()))
            else:
                feature_df[col_name] = grouped_df[self.var_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            self.feature_cols.append(col_name)
            # Rolling Std:
            if include_std_dev:
                col_name = f'rolling_std_{window}'
                if apply_log_transformation:
                    feature_df[col_name] = np.log1p(grouped_df[self.var_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().fillna(0)))
                else:
                    feature_df[col_name] = grouped_df[self.var_col].transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).std().fillna(0))
                self.feature_cols.append(col_name)
        
        # Future Targets:
        self.target_cols = []
        for i in range(0, num_future_targets):
            col_name = f'target_t_plus_{i}'
            feature_df[col_name] = grouped_df[self.var_col].shift(-i)
            self.target_cols.append(col_name)
        
        # Remove rows with missing features:
        feature_df = feature_df[feature_df[self.feature_cols].isna().sum(axis=1) == 0]
        
        # Remove excessive rows with all 0s from the training data:
        mask = (feature_df[self.feature_cols].sum(axis=1) == 0) \
             & (feature_df[self.target_cols].fillna(0).sum(axis=1) == 0)
        if self.test_start_period is not None:
            mask = mask & (feature_df[self.period_col] < self.test_start_period)
        zero_rows = feature_df[mask].sample(frac=0.1, random_state=42)
        non_zero_rows = feature_df[~mask]
        feature_df = pd.concat([zero_rows, non_zero_rows])
        self.df = feature_df

    def _create_tmp_col_for_train(self, col_name : str):
        tmp_col = f'tmp_{col_name}'
        if self.test_start_period is not None:
            self.df[tmp_col] = np.where(self.df[self.period_col] >= self.test_start_period, np.nan, self.df[self.var_col])
        else:
            self.df[tmp_col] = self.df[self.var_col].copy()
        return tmp_col
    
    def _add_overdispersion_flag(self, grouped : DataFrameGroupBy):
        tmp_col = self._create_tmp_col_for_train(self.var_col)
        mean = grouped[tmp_col].transform('mean')
        var = grouped[tmp_col].transform('var')
        self.df['overdispersion'] = (var > mean).astype(int)
        self.df.drop(columns=[tmp_col], inplace=True)

    def _fit_distributions(self, grouped : DataFrameGroupBy, use_best_fit_feature : bool = False, one_hot : bool = True):
        tmp_col = self._create_tmp_col_for_train(self.var_col)
        self.logger.info("fitting distributions...")
        
        def distr_fit(group_df: pd.DataFrame) -> pd.Series:
            data = group_df[tmp_col].dropna().to_numpy(dtype=int)
            best_fit, best_fit_params = compute_best_fit_distribution(data)
            # Pad parameters to a fixed length (3) for contiguous float arrays
            params = list(best_fit_params) + [np.nan] * (3 - len(best_fit_params))
            return pd.Series({
                'best_fit': best_fit,
                'distr_param_0': params[0],
                'distr_param_1': params[1],
                'distr_param_2': params[2]
            })
        # Compute parameters once per group, then merge back (much lower memory footprint)
        distr_df = grouped[[tmp_col]].apply(distr_fit).reset_index()
        self.df = pd.merge(self.df, distr_df, on=self.ts_id_cols, how='left')
        self.df['best_fit'] = self.df['best_fit'].astype('category')
        
        self.df.drop(columns=[tmp_col], inplace=True)
        self.logger.info("distributions fitting completed")

        if use_best_fit_feature:
            if one_hot:
                dummies = pd.get_dummies(self.df['best_fit'], prefix='best_fit', dtype=int, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.feature_cols.extend([col for col in self.df.columns if col.startswith(f'best_fit_')])
            else:
                self.categorical_feature_cols.append('best_fit')

    def add_calendar_features(self, df_calendar : pd.DataFrame, day_of_week_col : str, use_week_end_feature : bool = False, use_day_of_week_features : bool = True):
        if use_week_end_feature and use_day_of_week_features:
            raise ValueError("Only one of 'use_week_end_feature' or 'use_day_of_week_features' can be True")
        df_calendar[self.period_col] = df_calendar[self.period_col].astype('category')
        df_calendar[day_of_week_col] = df_calendar[day_of_week_col].astype('category')
        self.df = pd.merge(self.df, df_calendar[[self.period_col, day_of_week_col]], on=self.period_col, how='left')
        if use_week_end_feature:
            self._add_weekend_feature(day_of_week_col)
        if use_day_of_week_features:
            self._add_day_of_week_features(day_of_week_col, self.use_one_hot_encoding)
    
    def _add_weekend_feature(self, day_of_week_col : str = None):
        if day_of_week_col is None:
            self.df['is_weekend'] = np.where(pd.to_datetime(self.df[self.period_col]).dt.dayofweek.isin({5, 6}), 1, 0)
        else:
            # Saturdays and Sundays are marked as 1 and 2, respectively in the M5 data set:
            self.df['is_weekend'] = np.where(self.df[day_of_week_col].isin({1, 2}), 1, 0)
        self.feature_cols.append('is_weekend')
    
    def _add_day_of_week_features(self, day_of_week_col : str, one_hot : bool = True):
        if day_of_week_col is None:
            day_of_week_col = 'wday'
            self.df['wday'] = pd.to_datetime(self.df[self.period_col]).dt.dayofweek
        if one_hot:
            self.df = pd.get_dummies(self.df, columns=[day_of_week_col], dtype=int, drop_first=True)
            self.feature_cols.extend([col for col in self.df.columns if col.startswith(f'{day_of_week_col}_')])
        else:
            self.df[day_of_week_col] = self.df[day_of_week_col].astype('category')
            self.categorical_feature_cols.append(day_of_week_col)
        
    def add_feature(self, df_feature : pd.DataFrame, feature_name : str, join_on : list[str], is_cat_feature : bool = False, default_value : Any = np.nan, lags : tuple[int] | None = None, rolling_windows : tuple[int] | None = None):
        df_feature = df_feature.copy()
        df_feature[self.period_col] = df_feature[self.period_col].astype('category')
        for col in join_on:
            df_feature[col] = df_feature[col].astype('category')
        keys = join_on + [self.period_col]
        df_feature = df_feature[keys + [feature_name]].drop_duplicates()
        self.df = pd.merge(self.df, df_feature, on=keys, how='left') #.fillna({feature_name : default_value})

        self.df.sort_values(by=self.ts_id_cols + [self.period_col], inplace=True)
        gdf = self.df.groupby(self.ts_id_cols)
        self.df[feature_name] = gdf[feature_name].ffill().bfill().fillna(default_value)
        
        # Add lags if required:
        cols_lags = []
        if lags is not None:
            for lag in lags:
                col_name = f'{feature_name}_lag_{lag}'.replace('-', 'minus_')
                self.df[col_name] = gdf[feature_name].shift(lag)
                cols_lags.append(col_name)
        
        # Add rolling means if required:
        cols_rolling = []
        if (rolling_windows is not None) & (not is_cat_feature):
            for window in rolling_windows:
                col_name = f'{feature_name}_rolling_mean_{window}'
                self.df[col_name] = gdf[feature_name].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                cols_rolling.append(col_name)

        generated_cols = cols_lags + cols_rolling
        for col in generated_cols:
            self.df[col] = gdf[col].ffill().bfill()
        
        feature_list = [feature_name] + generated_cols
        if is_cat_feature and self.use_one_hot_encoding:
            original_cols = self.df.columns.copy()
            self.df = pd.get_dummies(self.df, columns=feature_list, dtype=int, drop_first=True)
            new_cols = self.df.columns.difference(original_cols)
            self.feature_cols.extend(new_cols)
        elif is_cat_feature and not self.use_one_hot_encoding:
            for f in feature_list:
                self.df[f] = self.df[f].astype('category')
                self.categorical_feature_cols.append(f)
        else:
            self.feature_cols.extend(feature_list)

    def train_glm(self, train_groups : tuple[str] = None) -> pd.DataFrame:
        partition_cols = list(train_groups) if train_groups is not None else []
        if 'overdispersion' not in partition_cols:
            partition_cols.append('overdispersion')

        if self.test_start_period is not None:
            train_df = self.df[self.df['is_test'] == 0]
        else:
            train_df = self.df
        
        def _train_group(group_df: pd.DataFrame) -> pd.Series:
            group_name = getattr(group_df, 'name', 'unknown')
            # Extract overdispersion from the group name instead of the dataframe
            if isinstance(group_name, tuple):
                idx = partition_cols.index('overdispersion')
                is_overdispersed = group_name[idx]
            else:
                is_overdispersed = group_name
                
            model_type = 'nbinom' if is_overdispersed == 1 else 'poisson'
            
            group_result = {}
            for target in self.target_cols:
                self.logger.info(f"training model for group ({group_name}), target {target}...")
                model_data, summary = train_glm(group_df, self.feature_cols, target, model_type)
                group_result[target] = model_data
                self.logger.info(summary)
                self.logger.info("model trained successfully")
            
            return pd.Series(group_result)

        try:
            # Opt into future behavior to silence the warning (pandas >= 2.2.0)
            result_df = train_df.groupby(partition_cols).apply(_train_group, include_groups=False).reset_index()
        except TypeError:
            # Graceful fallback for older pandas versions
            result_df = train_df.groupby(partition_cols).apply(_train_group).reset_index()
        return result_df
    
    def predict_glm(self, model_data : pd.DataFrame) -> pd.DataFrame:
        test_data = self.df[self.df['is_test'] == 1].copy()
        train_groups = [col for col in model_data.columns if col not in self.target_cols]
        
        # Initialize prediction columns
        for target in self.target_cols:
            test_data[f'{target}_mu'] = np.nan
            test_data[f'{target}_alpha'] = np.nan

        # Convert model_data to a lookup dictionary for O(1) model retrieval
        model_dict_map = model_data.set_index(train_groups).to_dict(orient='index')
        for group_keys, group_df in test_data.groupby(train_groups):
            if len(group_keys) == 1:
                group_keys = group_keys[0]
            if group_keys not in model_dict_map:
                continue
            row_models = model_dict_map[group_keys]
            group_idx = group_df.index
            # Prepare feature matrix and add Intercept
            X = group_df[self.feature_cols].copy()
            X['Intercept'] = 1.0
            for target in self.target_cols:
                model_dict = row_models.get(target)
                if not isinstance(model_dict, dict) or 'coefficients' not in model_dict:
                    continue
                coef_series = pd.Series(model_dict['coefficients'])
                # Align feature matrix with coefficients and safely evaluate dot product
                X_aligned = X.reindex(columns=coef_series.index).fillna(0.0)
                z = X_aligned.dot(coef_series)
                # Fetch target predictions
                test_data.loc[group_idx, f'{target}_mu'] = np.exp(z)
                test_data.loc[group_idx, f'{target}_alpha'] = model_dict.get('alpha', coef_series.get('alpha', np.nan))
        
        self._compute_forecasting_metrics(test_data, [f'{target}_mu' for target in self.target_cols])
        return test_data

    def _compute_forecasting_metrics(self, pred_df : pd.DataFrame, pred_columns : list[str]):
        if len(self.target_cols) != len(pred_columns):
            raise ValueError(f"Number of pred columns ({len(pred_columns)}) does not match the number of target columns ({len(self.target_cols)}).")
        
        target_to_pred = dict(zip(sorted(self.target_cols), sorted(pred_columns)))
        tmp_cols = []
        for col_name in self.target_cols:
            tmp_col = f'tmp_{col_name}'
            pred_df[tmp_col] = np.where(pred_df[col_name].isna(), 0, pred_df[target_to_pred[col_name]])
            tmp_cols.append(tmp_col)
        
        pred_df['actual'] = pred_df[self.target_cols].fillna(0).sum(axis=1)
        pred_df['pred'] = pred_df[tmp_cols].sum(axis=1)
        pred_df.drop(columns=tmp_cols, inplace=True)
        pred_df['under_diff'] = np.maximum(pred_df['actual'] - pred_df['pred'], 0)
        pred_df['over_diff'] = np.maximum(pred_df['pred'] - pred_df['actual'], 0)
        pred_df['abs_diff'] = pred_df['under_diff'] + pred_df['over_diff']

    def compute_lead_time_demand_glm(self, gdf : pd.DataFrame, lead_time : DiscreteDistribution, cycle_time : int, glm_mu_cols : list[str] = None, glm_alpha_cols : list[str] = None, return_distribution: bool = False, pred_first_day : bool = True) -> pd.DataFrame:
        if pred_first_day:
            first_day = gdf[self.period_col].min()
            gdf = gdf[gdf[self.period_col] == first_day].copy()
        else:
            gdf = gdf.copy()
        if glm_mu_cols is None:
            glm_mu_cols = [f"{col_name}_mu" for col_name in self.target_cols]
        if glm_alpha_cols is None:
            glm_alpha_cols = [f"{col_name}_alpha" for col_name in self.target_cols]
        horizon = len(glm_mu_cols)
        if horizon != len(glm_alpha_cols):
            raise ValueError("Number of 'mu' and 'alpha' columns do not match.")
        if horizon == 0:
            raise ValueError("At least one GLM forecast horizon is required.")
        if cycle_time < 0:
            raise ValueError("'cycle_time' must be non-negative.")
        
        if not return_distribution:
            service_levels_arr = gdf['service_level'].to_numpy(dtype=float)
            if np.any((service_levels_arr < 0) | (service_levels_arr > 1)):
                raise ValueError("'service_levels' values must be between 0 and 1.")

        param_cols = ['best_fit', 'distr_param_0', 'distr_param_1', 'distr_param_2']
        required_cols = glm_mu_cols + glm_alpha_cols + param_cols
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        if missing_cols:
            raise ValueError(f"Missing columns required to compute target inventory: {missing_cols}.")
        
        lead_time_with_cycle = lead_time.copy()
        lead_time_with_cycle.shift(cycle_time)
        if lead_time_with_cycle.min < 0:
            raise ValueError("Lead time plus cycle time must have non-negative support.")
        
        results = []
        for row_num, data_tuple in enumerate(gdf[required_cols].itertuples(index=False, name=None)):
            mu = data_tuple[:horizon]
            alpha = data_tuple[horizon:(2*horizon)]
            best_fit = data_tuple[2*horizon]
            best_fit_params = tuple(p for p in data_tuple[(2*horizon + 1):] if not pd.isna(p))

            default_demand = None
            if lead_time_with_cycle.max > horizon:
                default_demand = DiscreteDistribution.from_parametric(best_fit, best_fit_params)

            demand = []
            for (curr_mu, curr_alpha) in zip(mu, alpha):
                if pd.isna(curr_mu) or curr_mu < 0:
                    raise ValueError(f"Invalid GLM mean for row {row_num}: {curr_mu}.")
                if pd.isna(curr_alpha) or curr_alpha <= 0:
                    demand.append(DiscreteDistribution.from_parametric('poisson', (float(curr_mu), 0)))
                else:
                    n = 1 / curr_alpha
                    p = 1 / (1 + curr_mu * curr_alpha)
                    demand.append(DiscreteDistribution.from_parametric('nbinom', (n, p, 0)))
            demand_during_lead_time = lead_time_with_cycle.random_sum(demand, default_demand)
            
            if return_distribution:
                results.append(demand_during_lead_time)
            else:
                results.append(demand_during_lead_time.quantile(service_levels_arr[row_num]).item())

        if return_distribution:
            gdf['demand_distribution'] = results
        else:
            gdf['target_inventory'] = results
        return gdf

def compute_lead_time_demand(ts : DiscreteTimeSeries, model_data : pd.DataFrame, lead_time : dict[Any, DiscreteDistribution], lead_time_keys : list[str], cycle_time : int, service_levels : pd.DataFrame | None = None, model_type : str = 'glm', return_distribution: bool = False, pred_first_day: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    if model_type == 'glm':
        predictions = ts.predict_glm(model_data)
        if not return_distribution:
            if service_levels is not None:
                predictions = pd.merge(predictions, service_levels, on=[col for col in service_levels.columns if col != 'service_level'], how='left').fillna({'service_level': 0.95})
            else:
                predictions['service_level'] = 0.95
        cols = predictions.columns.to_list()
        predictions = predictions.groupby(lead_time_keys, as_index=False)[cols].apply(lambda x: ts.compute_lead_time_demand_glm(x, lead_time[x.name], cycle_time, return_distribution=return_distribution, pred_first_day=pred_first_day))
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'. Supported options are 'glm'.")
    return predictions

def compute_distribution_dict(df : pd.DataFrame, value : str = 'lead_time', keys : list[str] = None, method : str = 'actual') -> tuple[dict[Any, "DiscreteDistribution"], list[str]]:
    if keys is None:
        keys = [col for col in df.columns if col != value]
    distr = df.groupby(keys)[value].apply(lambda x: DiscreteDistribution.from_data(x, method)).to_dict()
    return distr, keys
