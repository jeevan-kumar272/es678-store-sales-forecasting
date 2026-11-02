"""Model definitions. Every learned step (imputation, scaling, encoding) lives inside a Pipeline,
so each CV fold fits it on its own training rows only. All models predict log1p(Sales) and are
converted back to euros."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import lightgbm as lgb

CAT = ["dow", "month", "holiday_code", "store_type", "assortment"]
B1_NUM = ["promo", "lag_7", "roll_mean_28", "store_mean_past"]       # the small baseline model (B1)
MODELS = ["linear", "ridge", "lasso", "random_forest", "lightgbm"]

def _num_pipe():
    return Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])

def _log(reg):
    return TransformedTargetRegressor(regressor=reg, func=np.log1p, inverse_func=np.expm1)

def _linear_pre(features):
    cat = [c for c in CAT if c in features]; num = [c for c in features if c not in cat]
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", _num_pipe(), num)])

def make_model(name, features, seed=42, params=None):
    """Returns (estimator, list_of_input_columns)."""
    p = dict(params or {})
    if name == "linear":                                             # B1: plain least squares, few features
        cols = ["dow"] + B1_NUM
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), ["dow"]), ("num", _num_pipe(), B1_NUM)])
        return _log(Pipeline([("pre", pre), ("reg", LinearRegression())])), cols
    if name == "ridge":
        return _log(Pipeline([("pre", _linear_pre(features)), ("reg", Ridge(alpha=p.get("alpha", 10.0)))])), list(features)
    if name == "lasso":
        return _log(Pipeline([("pre", _linear_pre(features)), ("reg", Lasso(alpha=p.get("alpha", 1e-3), max_iter=5000))])), list(features)
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
