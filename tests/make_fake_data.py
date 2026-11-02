"""Makes small fake train.csv and store.csv in data/fake so you can test code without Kaggle.
Usage: python tests/make_fake_data.py"""
import pathlib, numpy as np, pandas as pd
rng = np.random.default_rng(0)
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "fake"; out.mkdir(parents=True, exist_ok=True)
dates = pd.date_range("2013-01-01", "2015-07-31"); stores = np.arange(1, 61)
df = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"]).to_frame(index=False)
df["DayOfWeek"] = df.Date.dt.dayofweek + 1
df["Open"] = np.where(df.DayOfWeek == 7, 0, 1)
df["Promo"] = rng.integers(0, 2, len(df)); df["SchoolHoliday"] = rng.integers(0, 2, len(df))
df["StateHoliday"] = rng.choice(["0", "a", "b", "c"], len(df), p=[0.97, 0.015, 0.01, 0.005])
base = np.exp(rng.normal(8.5, 0.3, len(stores)))[df.Store.values - 1]
df["Sales"] = (base * (1 + 0.2 * df.Promo) * rng.lognormal(0, 0.15, len(df)) * df.Open).round().astype(int)
df["Customers"] = (df.Sales / 9 * rng.lognormal(0, 0.05, len(df))).round().astype(int)
gap = (df.Store.isin(range(1, 11))) & (df.Date >= "2014-07-01") & (df.Date <= "2014-12-31")
df = df[~gap]
df[["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].to_csv(out / "train.csv", index=False)
st = pd.DataFrame({"Store": stores, "StoreType": rng.choice(list("abcd"), len(stores)), "Assortment": rng.choice(list("abc"), len(stores)),
                   "CompetitionDistance": rng.integers(100, 20000, len(stores)).astype(float),
                   "CompetitionOpenSinceMonth": rng.integers(1, 13, len(stores)).astype(float), "CompetitionOpenSinceYear": rng.integers(2000, 2014, len(stores)).astype(float),
                   "Promo2": rng.integers(0, 2, len(stores)), "Promo2SinceWeek": np.nan, "Promo2SinceYear": np.nan, "PromoInterval": np.nan})
st.loc[:3, "CompetitionDistance"] = np.nan
st.to_csv(out / "store.csv", index=False)
print("Fake data written to", out)
