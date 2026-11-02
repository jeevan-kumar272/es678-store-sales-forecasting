"""Model definitions. Every learned step (imputation, scaling, encoding) lives inside a Pipeline,
so each CV fold fits it on its own training rows only. All models predict log1p(Sales) and are
converted back to euros."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import lightgbm as lgb

CAT = ["dow", "month", "holiday_code", "store_type", "assortment"]
B1_NUM = ["promo", "lag_7", "roll_mean_28", "store_mean_past"]       # the small baseline model (B1)
MODELS = ["linear", "ridge", "lasso", "random_forest", "lightgbm"]

# Columns measured in euros or metres are heavy tailed. The target is modelled on a log scale, so for the linear
# models these columns are put on a log scale too; otherwise the biggest stores are extrapolated far too high.
LOG_COLS = ["lag_7", "lag_14", "lag_21", "lag_28", "lag_364", "roll_mean_7", "roll_mean_28", "roll_mean_56",
            "store_mean_past", "promo_mean_56", "nonpromo_mean_56", "competition_distance"]
SIGNED_LOG_COLS = ["slope_28"]                                      # euros per day, can be negative

def _signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

def _num_pipe(transform=None):
    steps = [("log", FunctionTransformer(transform, validate=False))] if transform is not None else []
    return Pipeline(steps + [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])

def _log(reg):
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)

def _linear_pre(features, log_features=True):
    cat = [c for c in CAT if c in features]
    logc = [c for c in LOG_COLS if c in features] if log_features else []
    slog = [c for c in SIGNED_LOG_COLS if c in features] if log_features else []
    num = [c for c in features if c not in cat + logc + slog]
    parts = [("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", _num_pipe(), num)]
    if logc: parts.append(("logc", _num_pipe(np.log1p), logc))
    if slog: parts.append(("slog", _num_pipe(_signed_log), slog))
    return ColumnTransformer(parts)

def make_model(name, features, seed=42, params=None):
    """Returns (estimator, list_of_input_columns)."""
    p = dict(params or {})
    if name == "linear":                                             # B1: plain least squares, few features
        cols = ["dow"] + B1_NUM
        lf = p.get("log_features", True)
        logc = [c for c in B1_NUM if c in LOG_COLS] if lf else []
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["dow"]),
                                 ("num", _num_pipe(), [c for c in B1_NUM if c not in logc])] +
                                ([("logc", _num_pipe(np.log1p), logc)] if logc else []))
        return _log(Pipeline([("pre", pre), ("reg", LinearRegression())])), cols
    if name == "ridge":
        return _log(Pipeline([("pre", _linear_pre(features, p.get("log_features", True))), ("reg", Ridge(alpha=p.get("alpha", 10.0)))])), list(features)
    if name == "lasso":
        return _log(Pipeline([("pre", _linear_pre(features, p.get("log_features", True))), ("reg", Lasso(alpha=p.get("alpha", 1e-3), max_iter=5000))])), list(features)
    if name == "random_forest":
        rf = RandomForestRegressor(n_estimators=p.get("n_estimators", 100), min_samples_leaf=p.get("min_samples_leaf", 20),
                                   max_features=p.get("max_features", 0.5), max_samples=p.get("max_samples", 0.3),
                                   n_jobs=-1, random_state=seed)
        return _log(Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", rf)])), list(features)
    if name == "lightgbm":
        g = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.8,
                 subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1)
        g.update(p)
        return _log(lgb.LGBMRegressor(**g)), list(features)
    raise ValueError("unknown model " + name)
