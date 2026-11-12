"""Makes every figure for the paper and the repository from the saved results. Reads files only; fits and scores nothing.
Each figure is skipped with a message if its input file is missing.
Usage: python src/make_figures.py [--out paper/figures]"""
import argparse, json, pathlib, sqlite3
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from utils import load_config, ROOT

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"baseline_seasonal_naive": "Same weekday\nlast week", "baseline_wd_mean4": "Mean of last 4\nsame weekdays", "linear": "Linear\nregression",
         "ridge": "Ridge", "lasso": "Lasso", "lightgbm": "LightGBM"}
HOLIDAY = {0: "none", 1: "public\nholiday", 2: "Easter", 3: "Christmas"}

def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True); p = out / name
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("  wrote", p); return p

def fig_data(features, conn, out):
    f = features[["date", "sales", "promo", "dow"]]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].hist(np.log1p(f.sales), bins=60, color="#2A7F8E"); ax[0].set_xlabel("log(1 + sales)"); ax[0].set_ylabel("store-days"); ax[0].set_title("(a) Sales on open days, log scale")
    m = f.groupby(["dow", "promo"]).sales.mean().unstack()
    for c, lab, col in ((0, "no promotion", "#9aa5b1"), (1, "promotion", "#1F3A5F")):
        if c in m: ax[1].plot(m.index, m[c], marker="o", ms=3, label=lab, color=col)
    ax[1].set_xlabel("day of week (1 = Monday)"); ax[1].set_ylabel("mean sales (euros)"); ax[1].legend(frameon=False); ax[1].set_title("(b) Weekday and promotion")
    if conn is not None:
        s = pd.read_sql("SELECT d.full_date AS date, COUNT(*) AS stores FROM fact_daily_sales f JOIN dim_date d ON f.date_key = d.date_key GROUP BY d.full_date ORDER BY d.full_date", conn)
        ax[2].plot(pd.to_datetime(s.date), s.stores, color="#B5483A", lw=0.9); ax[2].set_ylabel("stores with a record"); ax[2].set_title("(c) Missing store-days (refurbishment)")
        ax[2].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); ax[2].tick_params(axis="x", rotation=30)
    else: ax[2].axis("off")
    fig.tight_layout(); return _save(fig, out, "fig_data.png")

def fig_results(boot, out):
    b = boot.set_index("model").loc[[m for m in NAMES if m in set(boot.model)]]
    fig, ax = plt.subplots(figsize=(3.4, 2.5)); y = np.arange(len(b))
    cols = ["#9aa5b1"] * 2 + ["#2A7F8E"] * (len(b) - 3) + ["#1F3A5F"]
    ax.barh(y, b.mae, xerr=[b.mae - b.mae_lo, b.mae_hi - b.mae], color=cols[:len(b)], ecolor="black", capsize=2)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[m] for m in b.index]); ax.invert_yaxis(); ax.set_xlabel("test MAE (euros), 95% interval")
    for i, (m, v) in enumerate(b.mae.items()): ax.text(v + 30, i, "%.0f" % v, va="center", fontsize=6.5)
    ax.set_xlim(0, b.mae_hi.max() * 1.18); fig.tight_layout(); return _save(fig, out, "fig_results.png")

def fig_intervals(cov, summ, out):
    rows = [("all", "all", summ["conformal"]["coverage"], summ["quantile_regression"]["coverage"], summ["n_test_rows"])]
    for sl, g, lab in [("store_type", i, "type %s" % "abcd"[i]) for i in range(4)] + [("promo", 0, "no promo"), ("promo", 1, "promo"),
                       ("holiday_code", 1, "public hol."), ("holiday_code", 2, "Easter")] + [("month", m, pd.Timestamp(str(m) + "-01").strftime("%b")) for m in sorted(cov[cov.slice == "month"].group.unique())]:
        c = cov[(cov.slice == sl) & (cov.group.astype(str) == str(g)) & (cov.method == "conformal")]
        q = cov[(cov.slice == sl) & (cov.group.astype(str) == str(g)) & (cov.method == "quantile_regression")]
        if len(c) and len(q): rows.append((lab, g, float(c.coverage.iloc[0]), float(q.coverage.iloc[0]), int(c.n.iloc[0])))
    labels = [r[0] if r[4] > 1000 else "%s\n(n=%d)" % (r[0], r[4]) for r in rows]; x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.4, label="conformal", color="#1F3A5F"); ax.bar(x + 0.2, [r[3] for r in rows], 0.4, label="quantile regr.", color="#9aa5b1")
    ax.axhline(summ["level"], color="#B5483A", ls="--", lw=0.9, label="nominal %.0f%%" % (100 * summ["level"]))
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=5.5); ax.set_ylabel("real coverage on test"); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6); fig.tight_layout(); return _save(fig, out, "fig_intervals.png")

def fig_ablation_horizon(abl, hor, out):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
    if abl is not None:
        a = abl[abl.group_removed != abl.group_removed.iloc[0]].sort_values("mae_increase"); y = np.arange(len(a))
        ax[0].barh(y, a.mae_increase, xerr=[a.mae_increase - a.increase_lo, a.increase_hi - a.mae_increase], color="#2A7F8E", ecolor="black", capsize=2)
        ax[0].set_yticks(y); ax[0].set_yticklabels(["%s (%d)" % (g.replace("_", " "), n) for g, n in zip(a.group_removed, a.features_removed)])
        ax[0].axvline(0, color="black", lw=0.6); ax[0].set_xlabel("increase in MAE when the group is removed (euros)"); ax[0].set_title("(a) Feature-group ablation, development folds")
    else: ax[0].axis("off")
    if hor is not None:
        ax[1].plot(hor.horizon_days, hor.mae, marker="o", color="#1F3A5F"); ax[1].set_xlabel("forecast horizon (days ahead)"); ax[1].set_ylabel("MAE (euros)")
        ax[1].set_xscale("log"); ax[1].set_xticks(hor.horizon_days); ax[1].set_xticklabels(hor.horizon_days); ax[1].set_title("(b) Error against horizon, development folds")
        for h, m in zip(hor.horizon_days, hor.mae): ax[1].annotate("%.0f" % m, (h, m), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6.5)
    else: ax[1].axis("off")
    fig.tight_layout(); return _save(fig, out, "fig_ablation_horizon.png")

def fig_residuals(tp, chosen, out):
    d = tp[tp.baseline_seasonal_naive.notna()].copy(); d["res"] = d.sales - d[chosen]; d["lres"] = np.log1p(d.sales) - np.log1p(d[chosen])
    d["dow"] = pd.to_datetime(d.date).dt.day_of_week + 1
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 4.4)); s = d.sample(min(len(d), 20000), random_state=0)
    ax[0, 0].scatter(s[chosen], s.res, s=1, alpha=0.15, color="#1F3A5F"); ax[0, 0].axhline(0, color="#B5483A", lw=0.8)
    ax[0, 0].set_xlabel("predicted sales (euros)"); ax[0, 0].set_ylabel("residual (actual - predicted)"); ax[0, 0].set_title("(a) Residuals against fitted values")
    ax[0, 1].boxplot([d.res[d.dow == k] for k in range(1, 8)], showfliers=False)       # no label keyword: its name changed between matplotlib versions
    ax[0, 1].set_xticks(range(1, 8)); ax[0, 1].set_xticklabels([str(k) for k in range(1, 8)]); ax[0, 1].axhline(0, color="#B5483A", lw=0.8)
    ax[0, 1].set_xlabel("day of week (1 = Monday)"); ax[0, 1].set_ylabel("residual (euros)"); ax[0, 1].set_title("(b) Residuals by weekday")
    (osm, osr), (slope, icpt, _) = stats.probplot(d.lres.to_numpy(), dist="norm")
    ax[1, 0].plot(osm, osr, ".", ms=1.5, color="#2A7F8E"); ax[1, 0].plot(osm, slope * np.asarray(osm) + icpt, color="#B5483A", lw=0.8)
    ax[1, 0].set_xlabel("normal quantiles"); ax[1, 0].set_ylabel("log residual"); ax[1, 0].set_title("(c) Normal QQ plot of log residuals")
    daily = d.groupby("date").res.mean(); daily.index = pd.to_datetime(daily.index)
    ax[1, 1].plot(daily.index, daily.values, color="#9aa5b1", lw=0.6); ax[1, 1].plot(daily.index, daily.rolling(7, min_periods=3).mean(), color="#1F3A5F", lw=1)
    ax[1, 1].axhline(0, color="#B5483A", lw=0.8); ax[1, 1].set_ylabel("mean residual per day (euros)"); ax[1, 1].set_title("(d) Mean error over the test period"); ax[1, 1].tick_params(axis="x", rotation=30)
    fig.tight_layout(); return _save(fig, out, "fig_residuals.png")

def fig_cases(tp, chosen, worst, out):
    stores = list(worst.store_key.head(2)); fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for a, s in zip(ax, stores):
        d = tp[tp.store_key == s].sort_values("date"); x = pd.to_datetime(d.date)
        a.plot(x, d.sales, color="black", lw=0.8, label="actual"); a.plot(x, d[chosen], color="#2A7F8E", lw=0.8, label="forecast")
        a.set_title("Store %d" % s); a.set_ylabel("sales (euros)"); a.tick_params(axis="x", rotation=30); a.legend(frameon=False)
    fig.tight_layout(); return _save(fig, out, "fig_cases.png")

def main():
    cfg = load_config(); P = ROOT / "data" / "processed"; A = ROOT / "artifacts"
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(P / ("features_h%d.pkl" % cfg["horizon_days"]))); ap.add_argument("--test-preds", default=str(P / "test_preds.pkl"))
    ap.add_argument("--db", default=str(ROOT / cfg["data"]["db_path"])); ap.add_argument("--artifacts", default=str(A))
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures"))
    a = ap.parse_args(); A = pathlib.Path(a.artifacts); out = pathlib.Path(a.out)
    have = lambda p: pathlib.Path(p).exists()
    chosen = json.load(open(A / "model_choice.json"))["chosen"] if have(A / "model_choice.json") else "lightgbm"
    feats = pd.read_pickle(a.features) if have(a.features) else None
    conn = sqlite3.connect(a.db) if have(a.db) else None
    tp = pd.read_pickle(a.test_preds) if have(a.test_preds) else None
    if feats is not None: fig_data(feats, conn, out)
    else: print("  skipped fig_data (no features file)")
    if have(A / "test_bootstrap.csv"): fig_results(pd.read_csv(A / "test_bootstrap.csv"), out)
    else: print("  skipped fig_results")
    if have(A / "intervals_coverage_by_group.csv") and have(A / "intervals_summary.json"):
        fig_intervals(pd.read_csv(A / "intervals_coverage_by_group.csv"), json.load(open(A / "intervals_summary.json")), out)
    else: print("  skipped fig_intervals")
    abl = pd.read_csv(A / "ablation.csv") if have(A / "ablation.csv") else None
    hor = pd.read_csv(A / "horizon_sweep.csv") if have(A / "horizon_sweep.csv") else None
    if abl is not None or hor is not None: fig_ablation_horizon(abl, hor, out)
    else: print("  skipped fig_ablation_horizon")
    if tp is not None:
        fig_residuals(tp, chosen, out)
        if have(A / "audit_worst_stores.csv"): fig_cases(tp, chosen, pd.read_csv(A / "audit_worst_stores.csv"), out)
    else: print("  skipped fig_residuals and fig_cases (no test predictions)")

if __name__ == "__main__":
    main()
