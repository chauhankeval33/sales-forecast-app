# 📈 Drug Stores – Store-wise Sales Forecasting

A Streamlit web app that forecasts daily sales for each drug store up to 7 days ahead.

🔗 **Live app:** https://sales-forecast-app-8vrsjnfy6bbcyzhmtb5vtm.streamlit.app

## Features
- Forecast 1–7 days ahead for one or many stores
- Backtest on past dates and compare with actual sales
- Chart per store and CSV download of the forecast
- Optional upload of the real promo / holiday calendar

## Model
- Ensemble of LightGBM, XGBoost and HistGradientBoosting
- Features: calendar, promotions, holidays, competition, and lag / rolling sales (7 to 364 days)
- Test performance: **RMSPE 11.2%**, **R² 0.925**

## Project structure
```
├── app.py                 # Streamlit interface
├── forecasting.py         # feature engineering + prediction logic
├── requirements.txt       # pinned dependencies
├── artifacts/             # trained model, recent history, store table
└── Store-wise Sales Forecasting.ipynb   # training notebook
```

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Data
Daily sales and store information for 1,115 drug stores, from 2013-01-01 to 2015-07-31.
Forecasts are available up to 2015-08-07.

