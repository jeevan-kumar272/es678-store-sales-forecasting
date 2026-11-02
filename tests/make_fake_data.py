"""Makes small fake train.csv and store.csv so code can be tested without Kaggle.
Command line:  python tests/make_fake_data.py     (writes to data/fake)
In tests:      from make_fake_data import make; make(folder, n_stores=30)"""
import pathlib, numpy as np, pandas as pd

def make(out, n_stores=60, seed=0):
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2013-01-01", "2015-07-31"); stores = np.arange(1, n_stores + 1)
    df = pd.MultiIndex.from_product([stores, dates], names=["Store", "Date"]).to_frame(index=False)
    df["DayOfWeek"] = df.Date.dt.day_of_week + 1
    df["Open"] = np.where(df.DayOfWeek == 7, 0, 1)
    df["Promo"] = rng.integers(0, 2, len(df)); df["SchoolHoliday"] = rng.integers(0, 2, len(df))
    df["StateHoliday"] = rng.choice(["0", "a", "b", "c"], len(df), p=[0.97, 0.015, 0.01, 0.005])
    base = np.exp(rng.normal(8.5, 0.3, len(stores)))[df.Store.values - 1]
    df["Sales"] = (base * (1 + 0.2 * df.Promo) * rng.lognormal(0, 0.15, len(df)) * df.Open).round().astype(int)
    df["Customers"] = (df.Sales / 9 * rng.lognormal(0, 0.05, len(df))).round().astype(int)
    open_idx = df.index[df.Open == 1]
    zero = rng.choice(open_idx, 5, replace=False)               # a few open days with zero sales, like the real data
    df.loc[zero, ["Sales", "Customers"]] = 0
    gap = df.Store.isin(range(1, max(2, n_stores // 6))) & (df.Date >= "2014-07-01") & (df.Date <= "2014-12-31")
    df = df[~gap]                                               # refurbishment-style gap
    cols = ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    df[cols].to_csv(out / "train.csv", index=False)
    st = pd.DataFrame({"Store": stores, "StoreType": rng.choice(list("abcd"), len(stores)), "Assortment": rng.choice(list("abc"), len(stores)),
                       "CompetitionDistance": rng.integers(100, 20000, len(stores)).astype(float),
                       "CompetitionOpenSinceMonth": rng.integers(1, 13, len(stores)).astype(float),
                       "CompetitionOpenSinceYear": rng.integers(2000, 2014, len(stores)).astype(float),
                       "Promo2": rng.integers(0, 2, len(stores)), "Promo2SinceWeek": np.nan, "Promo2SinceYear": np.nan, "PromoInterval": pd.Series([None] * len(stores), dtype=object)})
    p2 = st.Promo2 == 1
    st.loc[p2, "Promo2SinceWeek"] = 10.0; st.loc[p2, "Promo2SinceYear"] = 2011.0; st.loc[p2, "PromoInterval"] = "Jan,Apr,Jul,Oct"
    st.loc[:2, "CompetitionDistance"] = np.nan
    st.loc[3:5, ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]] = np.nan
    st.to_csv(out / "store.csv", index=False)
    return out

if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[1]
    print("Fake data written to", make(root / "data" / "fake"))
