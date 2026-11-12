"""Streamlit page for the frozen model.   Run:  streamlit run src/app.py
Shows the forecast and its 80 percent interval for a store and day, the surrounding weeks, and which features pushed the
forecast up or down. Everything is an association learned from past data, not a cause."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from serve import load_default

st.set_page_config(page_title="Store sales forecast", layout="wide")

@st.cache_resource
def get_forecaster():
    return load_default()

f = get_forecaster()
st.title("Seven-day-ahead store sales forecast")
st.caption("Model: %s, trained through %s. Replay mode: dates from %s to %s that the warehouse holds. Forecast origin is 7 days before the target day."
           % (f.art["name"], f.art["trained_through"], f.first, f.last))
c1, c2 = st.columns(2)
store = c1.number_input("Store number", 1, 1115, 262)
date = c2.date_input("Target day", value=pd.Timestamp(f.last), min_value=pd.Timestamp(f.first), max_value=pd.Timestamp(f.last)).strftime("%Y-%m-%d")
with st.expander("What-if: change the planned calendar values (describes the model's reaction, not what a promotion would cause)"):
    promo = st.selectbox("Promotion planned", [None, 0, 1], format_func=lambda v: "as recorded" if v is None else ("no" if v == 0 else "yes"))
    hol = st.selectbox("State holiday", [None, "0", "a", "b", "c"], format_func=lambda v: "as recorded" if v is None else {"0": "none", "a": "public", "b": "Easter", "c": "Christmas"}[v])
try:
    r = f.forecast(int(store), date, promo, hol)
except LookupError as e:
    st.warning(str(e)); st.stop()
m1, m2, m3 = st.columns(3)
m1.metric("Forecast (euros)", "{:,.0f}".format(r["forecast"]))
m2.metric("%d%% interval" % round(100 * r["interval_level"]), "{:,.0f} to {:,.0f}".format(r["interval_low"], r["interval_high"]))
m3.metric("Actual (euros)", "{:,.0f}".format(r["actual_sales"]), delta="{:,.0f}".format(r["actual_sales"] - r["forecast"]), delta_color="off")
h = f.history(int(store), date); x = pd.to_datetime(h.date)
fig, ax = plt.subplots(figsize=(9, 3))
ax.fill_between(x, h.lo, h.hi, color="#9aa5b1", alpha=0.4, label="80% interval"); ax.plot(x, h.forecast, color="#2A7F8E", label="forecast")
ax.plot(x, h.sales, color="black", lw=1, label="actual"); ax.axvline(pd.Timestamp(date), color="#B5483A", ls="--", lw=0.8)
ax.set_ylabel("sales (euros)"); ax.legend(frameon=False); fig.autofmt_xdate(); st.pyplot(fig)
st.subheader("What pushed this forecast up or down")
ex = f.explain(int(store), date)
if ex: st.dataframe(pd.DataFrame(ex).round(3), use_container_width=True)
else: st.write("Explanations are available for the LightGBM model only.")
