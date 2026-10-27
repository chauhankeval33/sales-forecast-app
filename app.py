import io
from datetime import timedelta

import pandas as pd
import streamlit as st

from forecasting import Forecaster, HORIZON

st.set_page_config(page_title="Store-wise Sales Forecasting", page_icon="📈", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_forecaster():
    return Forecaster()


fc = load_forecaster()
last = fc.last_date.date()

st.title("📈 Drug Stores – Store-wise Sales Forecasting")
st.caption(f"Forecast daily sales for 1–{HORIZON} days. Data available up to **{last}**; "
           f"forecasts can run until **{fc.max_forecast_date.date()}**.")

# ----------------------------- sidebar inputs ------------------------------ #
with st.sidebar:
    st.header("Inputs")
    stores = st.multiselect("Store Id(s)", fc.store_ids, default=[1])
    start = st.date_input("Start date", value=last + timedelta(days=1),
                          min_value=last - timedelta(days=41), max_value=last + timedelta(days=HORIZON))
    max_days = (last + timedelta(days=HORIZON) - start).days + 1
    max_days = max(1, min(HORIZON, max_days))
    if max_days == 1:
        days = 1
        st.caption("Only 1 day can be forecast from this start date.")
    else:
        days = st.slider("Number of days", 1, max_days, max_days)
    st.caption("A start date on/before the last data date is a **backtest** – actual sales are shown for comparison.")

    with st.expander("Optional: upload real calendar (CSV)"):
        st.caption("Columns: Store, Date (YYYY-MM-DD), Promo, SchoolHoliday, StateHoliday, Open. "
                   "If not uploaded, promo is assumed same as 2 weeks ago and open days same as last week.")
        up = st.file_uploader("future_info.csv", type="csv")
    run = st.button("Forecast", type="primary", width="stretch")

# ------------------------------ prediction -------------------------------- #
if run:
    if not stores:
        st.warning("Select at least one store.")
        st.stop()

    future_info = None
    if up is not None:
        try:
            future_info = pd.read_csv(up)
            missing = {"Store", "Date", "Promo", "SchoolHoliday", "StateHoliday", "Open"} - set(future_info.columns)
            if missing:
                st.error(f"Uploaded file is missing columns: {sorted(missing)}")
                st.stop()
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            st.stop()

    try:
        with st.spinner("Predicting..."):
            res = fc.predict_batch(stores, start, days, future_info)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    st.session_state["res"] = res

res = st.session_state.get("res")
if res is None:
    st.info("Choose store(s) and a date range in the sidebar, then click **Forecast**.")
    st.stop()

# ------------------------------- results ---------------------------------- #
total = res.PredictedSales.sum()
c1, c2, c3 = st.columns(3)
c1.metric("Stores", res.Store.nunique())
c2.metric("Days", res.Date.nunique())
c3.metric("Total predicted sales", f"{total:,.0f}")

st.subheader("Predicted sales (0 = store closed)")
wide = res.pivot(index="Date", columns="Store", values="PredictedSales")
wide.index = wide.index.date
st.dataframe(wide.style.format("{:,.0f}"), width="stretch")

st.subheader("Chart")
store_for_chart = st.selectbox("Store", sorted(res.Store.unique()))
one = res[res.Store == store_for_chart].set_index("Date").PredictedSales.rename("Forecast")
first = one.index.min()
actual = fc.history(store_for_chart, days=28, end=min(fc.last_date, first - pd.Timedelta(days=1))).rename("Actual (history)")
chart = pd.concat([actual, one], axis=1)
if first <= fc.last_date:       # backtest: also show the actuals for the forecast window
    real = fc.history(store_for_chart, days=60, end=fc.last_date)
    chart["Actual (same days)"] = real.reindex(chart.index)
st.line_chart(chart)

st.download_button("⬇️ Download forecast (CSV)", res.to_csv(index=False).encode(),
                   file_name="sales_forecast.csv", mime="text/csv")
