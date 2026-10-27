"""
forecasting.py  -  inference logic copied from the notebook (Section 5 + Section 9).
The Streamlit app imports this file; nothing here trains a model.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
HORIZON = 7
NA_COLS = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear", "Promo2SinceWeek", "Promo2SinceYear"]
MONTH = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Aug", 9: "Sept", 10: "Oct", 11: "Nov", 12: "Dec"}


# --------------------------------------------------------------------------- #
# Feature engineering  (identical to notebook Section 5)
# --------------------------------------------------------------------------- #
def add_features(data, holidays):
    d = data.copy()
    # calendar
    d["DayOfWeek"] = d.Date.dt.dayofweek + 1
    d["Month"], d["Day"], d["WeekOfYear"] = d.Date.dt.month, d.Date.dt.day, d.Date.dt.isocalendar().week.astype(int)
    # competition & Promo2
    year = d.Date.dt.year
    comp = (year - d.CompetitionOpenSinceYear) * 12 + (d.Month - d.CompetitionOpenSinceMonth)
    d["CompetitionOpenMonths"] = np.where(d.CompetitionOpenSinceYear > 0, comp, 0).clip(0)
    p2 = (year - d.Promo2SinceYear) * 52 + (d.WeekOfYear - d.Promo2SinceWeek)
    d["Promo2Weeks"] = np.where(d.Promo2 == 1, p2, 0).clip(0)
    d["IsPromo2Month"] = [int(m in iv) for m, iv in zip(d.Month.map(MONTH), d.PromoInterval)]
    # holiday proximity (capped at 30 days)
    dates = d.Date.values
    nxt = np.searchsorted(holidays, dates, side="left")
    prv = np.searchsorted(holidays, dates, side="right") - 1
    d["DaysToHoliday"] = np.where(nxt < len(holidays), (holidays[nxt.clip(max=len(holidays) - 1)] - dates) / np.timedelta64(1, "D"), 30).clip(max=30)
    d["DaysSinceHoliday"] = np.where(prv >= 0, (dates - holidays[prv.clip(min=0)]) / np.timedelta64(1, "D"), 30).clip(max=30)

    # Store x Date tables -> lags and rolling means
    table = lambda col: d.pivot(index="Date", columns="Store", values=col)
    wide = table("Sales")
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max()))
    promo, cust = table("Promo").reindex(wide.index), table("Customers").reindex(wide.index)
    base, promo7 = wide.shift(HORIZON), promo.shift(HORIZON)
    feats = {
        "Lag_7": base, "Lag_14": wide.shift(14), "Lag_21": wide.shift(21), "Lag_28": wide.shift(28),
        "Lag_364": wide.shift(364),
        "Roll_7": base.rolling(7, min_periods=1).mean(), "Roll_28": base.rolling(28, min_periods=1).mean(),
        "Roll_91": base.rolling(91, min_periods=1).mean(), "Std_28": base.rolling(28, min_periods=2).std(),
        "PromoLag_7": promo7,
        "PromoMean_28": base.where(promo7 == 1).rolling(28, min_periods=1).mean(),
        "NoPromoMean_28": base.where(promo7 == 0).rolling(28, min_periods=1).mean(),
        "Cust_Lag_7": cust.shift(HORIZON), "Cust_Roll_28": cust.shift(HORIZON).rolling(28, min_periods=1).mean(),
    }
    lag_df = None
    for name, w in feats.items():
        long = w.rename_axis("Date").reset_index().melt("Date", var_name="Store", value_name=name)
        lag_df = long if lag_df is None else lag_df.assign(**{name: long[name].values})
    d = d.merge(lag_df, on=["Date", "Store"], how="left")

    # combined features
    d["Roll_28_Promo"] = np.where(d.Promo == 1, d.PromoMean_28, d.NoPromoMean_28)
    d["Roll_28_Promo"] = d["Roll_28_Promo"].fillna(d.Roll_28)
    d["SalesPerCust"] = d.Roll_28 / d.Cust_Roll_28
    d["WeekdayMean_4"] = d[["Lag_7", "Lag_14", "Lag_21", "Lag_28"]].mean(axis=1)
    return d


# --------------------------------------------------------------------------- #
# Forecaster  (notebook Section 9, wrapped in a class so state is explicit)
# --------------------------------------------------------------------------- #
class Forecaster:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        bundle = joblib.load(artifact_dir / "sales_forecast_model.joblib")
        self.model = bundle["model"]
        self.k = bundle["k"]
        self.features = bundle["features"]
        self.holidays = bundle["holidays"]
        self.last_date = pd.Timestamp(bundle["last_date"])
        pq, pkl = artifact_dir / "history.parquet", artifact_dir / "history.pkl"
        self.df = pd.read_parquet(pq) if pq.exists() else pd.read_pickle(pkl)   # cleaned, open-day history
        self.store = pd.read_csv(artifact_dir / "store.csv")
        self.store_ids = sorted(self.store.Store.unique().tolist())

    # ---- helpers ---------------------------------------------------------- #
    @property
    def max_forecast_date(self):
        return self.last_date + pd.Timedelta(days=HORIZON)

    def history(self, store_id, days=28, end=None):
        end = self.last_date if end is None else pd.Timestamp(end)
        h = self.df[(self.df.Store == store_id) & (self.df.Date > end - pd.Timedelta(days=days)) & (self.df.Date <= end)]
        return h.set_index("Date").Sales

    def _predict(self, X):
        return self.k * self.model.predict(X)

    # ---- main entry point -------------------------------------------------- #
    def predict_batch(self, store_ids, start_date, days=1, future_info=None):
        store_ids = [int(s) for s in np.atleast_1d(store_ids)]
        unknown = set(store_ids) - set(self.store_ids)
        if unknown:
            raise ValueError(f"Unknown store id(s): {sorted(unknown)}")
        start = pd.Timestamp(start_date)
        if days > HORIZON or start + pd.Timedelta(days=days - 1) > self.max_forecast_date:
            raise ValueError(f"Forecast must be <= {HORIZON} days and end by {self.max_forecast_date.date()}")
        if start < self.last_date - pd.Timedelta(days=45):
            raise ValueError("Start date too far in the past for the history stored with the app.")

        df, store = self.df, self.store
        hist = df[df.Store.isin(store_ids) & (df.Date >= start - pd.Timedelta(days=400)) & (df.Date < start)]
        fut = pd.DataFrame([(s, d) for s in store_ids for d in pd.date_range(start, periods=days)], columns=["Store", "Date"])
        fut = fut.merge(store, on="Store", how="left").assign(StateHoliday="0")
        fut["PromoInterval"] = fut["PromoInterval"].fillna("None")
        fut[["CompetitionDistance"] + NA_COLS] = fut[["CompetitionDistance"] + NA_COLS].fillna(0)

        if future_info is None:   # Promo = 14 days ago; Open / SchoolHoliday = 7 days ago
            promo14 = hist[["Store", "Date", "Promo"]].assign(Date=lambda x: x.Date + pd.Timedelta(days=14))
            wk7 = hist[["Store", "Date", "SchoolHoliday"]].assign(Date=lambda x: x.Date + pd.Timedelta(days=7), Open=1)
            fut = fut.merge(promo14, on=["Store", "Date"], how="left").merge(wk7, on=["Store", "Date"], how="left")
            fut = fut.fillna({"Promo": 0, "SchoolHoliday": 0, "Open": 0})
        else:
            fi = future_info.copy()
            fi["Date"] = pd.to_datetime(fi["Date"])
            fut = fut.drop(columns="StateHoliday").merge(fi, on=["Store", "Date"], how="left")
            fut["StateHoliday"] = fut["StateHoliday"].fillna("0").astype(str)
            fut = fut.fillna({"Promo": 0, "SchoolHoliday": 0, "Open": 0})

        hol = np.union1d(self.holidays, fut.loc[fut.StateHoliday != "0", "Date"].values)
        feats = add_features(pd.concat([hist, fut], ignore_index=True), hol)
        feats = feats[feats.Date >= start].copy()
        feats["PredictedSales"] = np.where(feats.Open == 1, self._predict(feats[self.features]), 0).round(0)
        out = feats[["Store", "Date", "DayOfWeek", "Promo", "Open", "PredictedSales"]]
        return out.sort_values(["Store", "Date"]).reset_index(drop=True)
