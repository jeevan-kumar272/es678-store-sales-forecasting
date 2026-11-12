"""Prediction API for the frozen model.
It loads artifacts/model.pkl, the conformal offsets, and the feature table for the loaded warehouse period. For a store and a
target date it returns the point forecast, an 80 percent interval, the forecast origin (target minus the horizon), the model
version, and, because the date is in the loaded period, the actual sales for comparison ("replay" mode).
Planned promotion, holiday and school-holiday values can be overridden for what-if checks; those results describe how the
model reacts, not what a promotion would cause.
Not built: forecasts for dates beyond the loaded data (that needs a calendar extension).
Run:  uvicorn serve:app --app-dir src --port 8000      Docs:  http://127.0.0.1:8000/docs"""
import json, pathlib
import joblib, numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from utils import load_config, ROOT
from intervals import apply_offsets

HOL = {"0": 0, "a": 1, "b": 2, "c": 3}

class ForecastRequest(BaseModel):
    store_id: int = Field(ge=1, le=1115, description="Rossmann store number")
    target_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="trading day to forecast, YYYY-MM-DD")
    promo: int | None = Field(default=None, ge=0, le=1, description="optional override: promotion planned on the target day")
    state_holiday: str | None = Field(default=None, pattern=r"^[0abc]$", description="optional override: 0 none, a public, b Easter, c Christmas")
    school_holiday: int | None = Field(default=None, ge=0, le=1, description="optional override: school holiday on the target day")

class Forecaster:
    def __init__(self, features_df, artifact, offsets_json, min_date="2015-01-01"):
        self.art = artifact; self.cols = artifact["columns"]
        self.offsets = {("all" if k == "all" else int(k)): tuple(v) for k, v in offsets_json["log_offsets"].items()}
        self.level = offsets_json["level"]
        self.df = features_df[features_df["date"] >= min_date].reset_index(drop=True)
        p = artifact["model"].predict(self.df[self.cols])
        self.df["forecast"] = p
        self.df["lo"] = apply_offsets(p, self.df.store_type, self.offsets, 0)
        self.df["hi"] = apply_offsets(p, self.df.store_type, self.offsets, 1)
        self.index = {(int(s), d): i for i, (s, d) in enumerate(zip(self.df.store_key, self.df.date))}
        self.first, self.last = self.df.date.min(), self.df.date.max()

    def _row(self, store_id, target_date):
        i = self.index.get((int(store_id), target_date))
        if i is None:
            raise LookupError("No forecast for store %s on %s. Supported: open trading days from %s to %s for stores that have a record. "
                              "Closed days and days outside the loaded period cannot be forecast." % (store_id, target_date, self.first, self.last))
        return i

    def forecast(self, store_id, target_date, promo=None, state_holiday=None, school_holiday=None):
        i = self._row(store_id, target_date); r = self.df.iloc[[i]].copy(); applied = {}
        if promo is not None: r["promo"] = promo; applied["promo"] = promo
        if state_holiday is not None: r["holiday_code"] = HOL[state_holiday]; applied["state_holiday"] = state_holiday
        if school_holiday is not None: r["school"] = school_holiday; applied["school_holiday"] = school_holiday
        if applied:
            p = float(self.art["model"].predict(r[self.cols])[0])
            lo = float(apply_offsets([p], r.store_type, self.offsets, 0)[0]); hi = float(apply_offsets([p], r.store_type, self.offsets, 1)[0])
        else:
            p, lo, hi = float(r.forecast.iloc[0]), float(r.lo.iloc[0]), float(r.hi.iloc[0])
        origin = (pd.Timestamp(target_date) - pd.Timedelta(days=self.art["horizon_days"])).strftime("%Y-%m-%d")
        return {"store_id": int(store_id), "target_date": target_date, "forecast": round(p, 1), "interval_low": round(lo, 1),
                "interval_high": round(hi, 1), "interval_level": self.level, "forecast_origin": origin,
                "history_used_up_to": origin, "model": self.art["name"], "model_trained_through": self.art["trained_through"],
                "overrides_applied": applied, "mode": "replay", "actual_sales": float(self.df.sales.iloc[i])}

    def history(self, store_id, target_date, days=28):
        """Forecasts, intervals and actuals for the days around the target, for plotting."""
        t = pd.Timestamp(target_date); lo, hi = (t - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), (t + pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        d = self.df[(self.df.store_key == store_id) & (self.df.date >= lo) & (self.df.date <= hi)]
        return d[["date", "sales", "forecast", "lo", "hi"]].sort_values("date")

    def explain(self, store_id, target_date, top=8):
        """Which features pushed this forecast up or down (LightGBM contributions, in log-sales units). Association, not cause."""
        inner = getattr(self.art["model"], "regressor_", None)
        if inner is None or not hasattr(inner, "booster_"): return []
        r = self.df.iloc[[self._row(store_id, target_date)]]
        c = inner.booster_.predict(r[self.cols], pred_contrib=True)[0]
        out = [{"feature": n, "log_effect": float(v), "approx_pct_effect": float(100 * (np.exp(v) - 1))} for n, v in zip(self.cols + ["baseline (average store-day)"], c)]
        return sorted(out, key=lambda x: -abs(x["log_effect"]))[:top]

def load_default(min_date="2015-01-01"):
    cfg = load_config(); A = ROOT / "artifacts"
    art = joblib.load(A / "model.pkl"); off = json.load(open(A / "conformal_offsets.json"))
    df = pd.read_pickle(ROOT / "data" / "processed" / ("features_h%d.pkl" % cfg["horizon_days"]))
    return Forecaster(df, art, off, min_date)

def create_app(forecaster=None):
    app = FastAPI(title="Store sales forecast", description="Seven-day-ahead daily sales forecast per Rossmann store, with an 80 percent interval.")
    state = {"f": forecaster}
    def get():
        if state["f"] is None: state["f"] = load_default()
        return state["f"]

    @app.get("/health")
    def health():
        f = get(); return {"status": "ok", "model": f.art["name"], "trained_through": f.art["trained_through"],
                           "supported_dates": [f.first, f.last], "horizon_days": f.art["horizon_days"], "mode": "replay"}

    @app.get("/schema")
    def schema():
        p = ROOT / "artifacts" / "schema.json"
        return json.load(open(p)) if p.exists() else ForecastRequest.model_json_schema()

    @app.post("/forecast")
    def forecast(req: ForecastRequest):
        try:
            return get().forecast(req.store_id, req.target_date, req.promo, req.state_holiday, req.school_holiday)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e))
    return app

app = create_app()
