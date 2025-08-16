"""
Forecasting Pro
No Code Forecasting App

Single-file Streamlit app.

This version:
- Uses Prophet, SARIMAX, Holt-Winters, GradientBoosting.
- Robust holdout metrics (MAE, RMSE, MAPE).
- Improved anomaly detection with narrative tooltips.
- Model comparison table with pill-style conditional formatting and a summary section that automatically highlights the best model by MAPE and explains why it's recommended.

Footer: Developed with ❤️Streamlit CE Innovation Labs 2025
"""

import streamlit as st
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import base64
from datetime import datetime, timedelta
import os

# forecasting libraries
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

import plotly.express as px
import plotly.graph_objects as go

# App metadata
APP_TITLE = "Forecasting Pro"
APP_SUBTITLE = "No Code Forecasting App"
FOOTER = "Developed with ❤️Streamlit CE Innovation Labs 2025"

# requirements content (written to requirements.txt)
REQUIREMENTS = r"""
streamlit>=1.20
pandas>=1.5
numpy>=1.23
plotly>=5.0
matplotlib>=3.5
prophet>=1.1
statsmodels>=0.13
scikit-learn>=1.1
lightgbm>=3.3
python-dateutil
"""

# ----------------- Helpers -----------------

def write_requirements_file():
    try:
        with open("requirements.txt", "w") as f:
            f.write(REQUIREMENTS)
    except Exception:
        pass


def set_page():
    st.set_page_config(page_title=f"{APP_TITLE} - {APP_SUBTITLE}", layout="wide")


def read_uploaded_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error("Could not read file. Please upload CSV or Excel.")
            raise e
    return df


def download_df(df: pd.DataFrame, filename: str):
    return st.download_button(label=f"Download {filename}", data=df.to_csv(index=False).encode('utf-8'), file_name=filename, mime='text/csv')

# Forecast functions

def forecast_with_prophet(df, date_col, value_col, periods, freq, seasonality_mode, yearly, weekly, daily, holidays_df=None):
    m = Prophet(seasonality_mode=seasonality_mode, yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily)
    if holidays_df is not None and not holidays_df.empty:
        m.holidays = holidays_df
    hist = df.rename(columns={date_col: 'ds', value_col: 'y'})[['ds', 'y']]
    m.fit(hist)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fcst = m.predict(future)
    return m, fcst


def sarimax_forecast(df, date_col, value_col, periods, freq, order=(1,0,0), seasonal_order=(0,0,0,0)):
    series = df.set_index(date_col)[value_col].asfreq(freq).fillna(method='ffill')
    model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=periods)
    idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    fcst = pd.DataFrame({'ds': idx, 'yhat': pred.predicted_mean, 'yhat_lower': pred.conf_int().iloc[:,0], 'yhat_upper': pred.conf_int().iloc[:,1]})
    return res, fcst


def holt_winters_forecast(df, date_col, value_col, periods, freq, seasonal='add', seasonal_periods=None):
    series = df.set_index(date_col)[value_col].asfreq(freq).fillna(method='ffill')
    n = len(series)
    if seasonal_periods is None:
        if freq == 'M':
            seasonal_periods = 12
        elif freq == 'D':
            seasonal_periods = 7
        elif freq == 'W':
            seasonal_periods = 52
        else:
            seasonal_periods = 12
    # fallback for short series
    if n < 2 * seasonal_periods:
        # warn user and try non-seasonal fit
        st.warning(f"Series length {n} is less than two seasonal cycles ({2*seasonal_periods}). Using non-seasonal ExponentialSmoothing fallback.")
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            res = model.fit(optimized=True)
            fcst = res.forecast(steps=periods)
            idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
            return res, pd.DataFrame({'ds': idx, 'yhat': fcst.values})
        except Exception:
            idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
            return None, pd.DataFrame({'ds': idx, 'yhat': np.repeat(series.iloc[-1], periods)})
    try:
        model = ExponentialSmoothing(series, trend='add', seasonal=seasonal, seasonal_periods=seasonal_periods)
        res = model.fit(optimized=True)
        fcst = res.forecast(steps=periods)
        idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
        return res, pd.DataFrame({'ds': idx, 'yhat': fcst.values})
    except Exception as e:
        st.warning(f"Holt-Winters failed: {e}. Falling back to non-seasonal.")
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal=None)
            res = model.fit(optimized=True)
            fcst = res.forecast(steps=periods)
            idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
            return res, pd.DataFrame({'ds': idx, 'yhat': fcst.values})
        except Exception:
            idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
            return None, pd.DataFrame({'ds': idx, 'yhat': np.repeat(series.iloc[-1], periods)})


def create_lag_features(series, lags=24):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    return df.dropna()


def ml_forecast_with_gb(df, date_col, value_col, periods, freq, lags=24):
    s = df.set_index(date_col)[value_col].asfreq(freq).fillna(method='ffill')
    lagged = create_lag_features(s, lags=lags)
    X = lagged.drop(columns=['y'])
    y = lagged['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    last_values = s.iloc[-lags:].tolist()
    preds = []
    for _ in range(periods):
        x_row = np.array(last_values[-lags:])[::-1].reshape(1, -1)
        p = model.predict(x_row)[0]
        preds.append(p)
        last_values.append(p)
    idx = pd.date_range(start=s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    return model, pd.DataFrame({'ds': idx, 'yhat': preds})


def detect_anomalies(series, contamination=0.01):
    iso = IsolationForest(contamination=contamination, random_state=42)
    X = series.values.reshape(-1,1)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)
    outliers = preds == -1
    return pd.Series(outliers, index=series.index), pd.Series(scores, index=series.index)


def explain_anomalies(df, anomaly_mask, categorical_cols, top_n=3):
    # align mask
    if isinstance(anomaly_mask, pd.Series):
        mask = anomaly_mask.reindex(df.index, fill_value=False)
    else:
        mask = pd.Series(anomaly_mask, index=df.index)
    anomalies = df[mask.astype(bool)]
    baseline = df[~mask.astype(bool)]
    explanations = []
    if anomalies.empty:
        return explanations
    for col in categorical_cols:
        if col not in df.columns:
            continue
        anom_counts = anomalies[col].value_counts(normalize=True).head(top_n)
        base_counts = baseline[col].value_counts(normalize=True).head(top_n)
        narrative = []
        for k,v in anom_counts.items():
            base_v = base_counts.get(k, 0)
            delta = (v - base_v) * 100
            if delta > 0:
                narrative.append(f"Category '{k}' appears {delta:.1f}% more often during anomalies")
            else:
                narrative.append(f"Category '{k}' not unusually frequent")
        explanations.append({'col': col, 'top_anom': anom_counts.to_dict(), 'top_base': base_counts.to_dict(), 'narrative': narrative})
    return explanations


def create_anomaly_tooltip(row, categorical_cols, explanations_map):
    parts = []
    for c in categorical_cols:
        if c in row and pd.notna(row[c]):
            parts.append(f"{c}: {row[c]}")
    narratives = []
    for col in categorical_cols:
        val = row.get(col)
        if val is None or pd.isna(val):
            continue
        key = f"{col}||{val}"
        if key in explanations_map:
            narratives.append(explanations_map[key])
    text = "<br>".join(parts + narratives)
    return text if text else "No categorical info"


def compute_holdout_metrics(series, model_name, predict_func, freq, test_frac=0.2):
    n = len(series)
    if n < 10:
        return None
    cutoff = int(n * (1 - test_frac))
    train = series.iloc[:cutoff]
    test = series.iloc[cutoff:]
    try:
        pred = predict_func(train, len(test))
        pred_series = pd.Series(pred.values if hasattr(pred, 'values') else pred, index=test.index)
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mae = mean_absolute_error(test, pred_series)
        rmse = float(np.sqrt(mean_squared_error(test, pred_series)))
        denom = test.replace(0, np.nan)
        mape = float(np.nanmean(np.abs((test - pred_series) / denom))) * 100
        return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    except Exception as e:
        return {'model': model_name, 'error': str(e)}

# ----------------- UI -----------------

def sidebar_file_upload():
    st.sidebar.header("Upload & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload time series CSV/Excel", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is None:
        st.sidebar.info("You can upload a CSV or Excel file with a date column and a numeric value column. Additional columns (categorical or numeric) can be used as exogenous features or for driver analysis.")
    return uploaded_file


def metrics_to_html_table(metrics_list):
    # returns an HTML table string with pill styling
    rows = []
    for m in metrics_list:
        model = m.get('model')
        if 'error' in m:
            mae = rmse = mape = 'error'
        else:
            mae = f"{m.get('MAE'):.3f}" if m.get('MAE') is not None else 'n/a'
            rmse = f"{m.get('RMSE'):.3f}" if m.get('RMSE') is not None else 'n/a'
            mape_val = m.get('MAPE')
            mape = f"{mape_val:.2f}%" if mape_val is not None else 'n/a'
        # pill color based on MAPE
        color = '#d3d3d3'
        if 'error' in m:
            color = '#cccccc'
        else:
            mv = m.get('MAPE')
            if mv is None or np.isnan(mv):
                color = '#cccccc'
            elif mv < 10:
                color = '#1aaf6c'  # green
            elif mv < 20:
                color = '#2da9ff'  # blue-ish (good)
            elif mv < 50:
                color = '#ffb020'  # orange
            else:
                color = '#ff4d4f'  # red
        row = f"<tr><td>{model}</td><td><span style='background:{color};padding:6px 10px;border-radius:999px;color:#fff'>{mape}</span></td><td>{mae}</td><td>{rmse}</td></tr>"
        rows.append(row)
    table = "<table style='border-collapse:collapse;width:100%'><thead><tr><th style='text-align:left;padding:8px'>Model</th><th style='text-align:left;padding:8px'>MAPE (pill)</th><th style='text-align:left;padding:8px'>MAE</th><th style='text-align:left;padding:8px'>RMSE</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return table


def pick_best_model(metrics_list):
    # choose lowest MAPE (numeric), ignore errors and nans
    best = None
    for m in metrics_list:
        if 'error' in m:
            continue
        mv = m.get('MAPE')
        if mv is None or np.isnan(mv):
            continue
        if best is None or mv < best.get('MAPE'):
            best = m
    return best


def interpret_metrics(metrics):
    # return simple narrative based on thresholds
    if metrics is None:
        return "Not enough data to compute metrics."
    if 'error' in metrics:
        return f"Could not compute metrics for {metrics.get('model')}: {metrics.get('error')}"
    mape = metrics.get('MAPE')
    mae = metrics.get('MAE')
    rmse = metrics.get('RMSE')
    parts = []
    parts.append(f"Model: **{metrics.get('model')}**")
    parts.append(f"MAPE: **{mape:.2f}%** — this measures average percent error.")
    if mape < 10:
        parts.append("Interpretation: Excellent forecast accuracy (green). Predictions are very close to actuals on average.")
    elif mape < 20:
        parts.append("Interpretation: Good accuracy (blue). Forecasts are reliable for many business uses.")
    elif mape < 50:
        parts.append("Interpretation: Moderate accuracy (orange). Use with caution; consider additional features or cross-validation.")
    else:
        parts.append("Interpretation: Poor accuracy (red). Model is not capturing the series well — try different models or more data.")
    parts.append(f"MAE: **{mae:.3f}** (lower is better) — absolute average error in target units.")
    parts.append(f"RMSE: **{rmse:.3f}** (lower is better) — penalizes larger errors more than MAE.")
    return "

".join(parts)


def main():
    set_page()
    write_requirements_file()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    tabs = st.tabs(["Data", "Forecast", "Anomaly & Drivers", "Model Compare", "Summary & Notes"])

    uploaded_file = sidebar_file_upload()

    if uploaded_file is None:
        st.info("No file uploaded — using a built-in example (monthly retail sales). Upload your own in the sidebar to run on your data.")
        dates = pd.date_range(start='2015-01-01', periods=120, freq='M')
        data = (np.sin(np.arange(120) / 6) + np.arange(120) / 20 + np.random.normal(0, 0.3, 120)) * 100 + 500
        df = pd.DataFrame({'ds': dates, 'y': data})
        df['store'] = np.random.choice(['A', 'B', 'C'], size=len(df))
    else:
        df = read_uploaded_file(uploaded_file)

    # Shared selection defaults
    with tabs[0]:
        st.header("Data Preview")
        st.markdown("**Upload format help:** The app expects a single date column and one numeric target column. Additional columns (categorical or numeric) can be used as exogenous features or for driver analysis.")
        st.dataframe(df.head(100))
        st.markdown("---")
        with st.expander("Map columns (date & value)"):
            cols = df.columns.tolist()
            date_col = st.selectbox("Date column", cols, index=0)
            value_col = st.selectbox("Value column", cols, index=1 if len(cols) > 1 else 0)
            freq = st.selectbox("Data frequency (if unknown pick 'infer')", ['infer', 'D', 'W', 'M', 'Q', 'H'])
            if freq == 'infer':
                try:
                    inferred = pd.infer_freq(pd.to_datetime(df[date_col]))
                    st.write(f"Inferred frequency: {inferred}")
                    freq_val = inferred if inferred is not None else 'D'
                except Exception:
                    st.write("Could not infer frequency; choose manually")
                    freq_val = 'D'
            else:
                freq_val = freq
            st.write("Sample range:")
            try:
                st.write(f"{pd.to_datetime(df[date_col]).min().date()} to {pd.to_datetime(df[date_col]).max().date()}")
            except Exception:
                st.write("Date parsing error — check your date column")

    # Forecast Tab
    with tabs[1]:
        st.header("Forecast")
        st.markdown("Choose model, set horizon, seasonality and holidays. Guidance notes are on the right.")
        with st.sidebar.expander("Forecast Settings", expanded=True):
            model_choice = st.selectbox("Model", ["Prophet", "SARIMAX", "Holt-Winters", "GradientBoosting"])
            periods = st.number_input("Forecast periods (horizon)", min_value=1, max_value=1000, value=12)
            user_freq = st.selectbox("Frequency (pandas offset) e.g. 'M' monthly, 'D' daily", ['M', 'D', 'W', 'H'])
            seasonality_mode = st.selectbox("Seasonality Mode (Prophet only)", ['additive', 'multiplicative'])
            yearly = st.checkbox("Yearly seasonality", value=True)
            weekly = st.checkbox("Weekly seasonality", value=False)
            daily = st.checkbox("Daily seasonality", value=False)
            include_holidays = st.checkbox("Include holidays CSV (two columns: ds, holiday)", value=False)
            if include_holidays:
                holidays_file = st.sidebar.file_uploader("Upload holidays CSV", type=['csv'])
            download_button = st.sidebar.checkbox("Provide download button for forecast CSV", value=True)

        working_df = df.copy()
        try:
            working_df[date_col] = pd.to_datetime(working_df[date_col])
        except Exception as e:
            st.error("Could not parse the selected date column to datetime. Check the file format.")
            st.exception(e)
            return

        run = st.button("Run Forecast")
        if run:
            with st.spinner("Fitting model..."):
                try:
                    if model_choice == 'Prophet':
                        if Prophet is None:
                            st.error("Prophet package not available in this environment. Please install 'prophet'.")
                            raise RuntimeError('Prophet missing')
                        holidays_df = None
                        if include_holidays and 'holidays_file' in locals() and holidays_file is not None:
                            try:
                                holidays_df = pd.read_csv(holidays_file)
                            except Exception:
                                st.error("Could not read holidays file; ensure two columns 'ds' and 'holiday'")
                        m, fcst = forecast_with_prophet(working_df, date_col, value_col, periods, user_freq, seasonality_mode, yearly, weekly, daily, holidays_df)
                        st.success("Forecast complete (Prophet)")
                        hist = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='history'))
                        if 'yhat' in fcst.columns:
                            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], mode='lines', name='forecast'))
                        elif 'y' in fcst.columns:
                            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['y'], mode='lines', name='forecast'))
                        if 'yhat_upper' in fcst.columns:
                            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_upper'], mode='lines', name='upper', line={'dash': 'dash'}, visible='legendonly'))
                        st.plotly_chart(fig, use_container_width=True)
                        if 'yhat' in fcst.columns:
                            out_df = fcst[['ds', 'yhat']].rename(columns={'yhat': 'y'})
                        elif 'y' in fcst.columns:
                            out_df = fcst[['ds', 'y']]
                        else:
                            out_df = fcst[['ds']].copy()

                    elif model_choice == 'SARIMAX':
                        order = (1,1,1)
                        seasonal_order = (1,1,1,12) if user_freq == 'M' else (1,1,1,7)
                        res, fcst = sarimax_forecast(working_df, date_col, value_col, periods, user_freq, order, seasonal_order)
                        st.success("Forecast complete (SARIMAX)")
                        hist = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='history'))
                        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='forecast'))
                        if 'yhat_upper' in fcst.columns:
                            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat_upper'], name='upper', line={'dash': 'dash'}, visible='legendonly'))
                        st.plotly_chart(fig, use_container_width=True)
                        out_df = fcst.rename(columns={'yhat': 'y'})[['ds', 'y']]

                    elif model_choice == 'Holt-Winters':
                        seasonal = st.selectbox("Seasonal type", ['add', 'mul'])
                        seasonal_periods = st.number_input("Seasonal periods (e.g. 12 for monthly)", min_value=1, max_value=366, value=12)
                        res, fcst = holt_winters_forecast(working_df, date_col, value_col, periods, user_freq, seasonal, seasonal_periods)
                        st.success("Forecast complete (Holt-Winters)")
                        hist = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='history'))
                        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='forecast'))
                        st.plotly_chart(fig, use_container_width=True)
                        out_df = fcst.rename(columns={'yhat': 'y'})[['ds', 'y']]

                    elif model_choice == 'GradientBoosting':
                        model, fcst = ml_forecast_with_gb(working_df, date_col, value_col, periods, user_freq)
                        st.success("Forecast complete (GradientBoosting)")
                        hist = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='history'))
                        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='forecast'))
                        st.plotly_chart(fig, use_container_width=True)
                        out_df = fcst.rename(columns={'yhat': 'y'})[['ds', 'y']]

                    st.markdown("**Forecast sample (first rows)**")
                    st.dataframe(out_df.head(20))
                    if download_button:
                        download_df(out_df, 'forecast.csv')

                    # holdout metrics
                    st.markdown("---")
                    st.subheader("Holdout accuracy estimate (simple) -> MAE, RMSE, MAPE")
                    try:
                        series = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                        metrics = None

                        def prophet_predict(train_series, h):
                            if Prophet is None:
                                raise RuntimeError('Prophet not available')
                            m = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily, seasonality_mode=seasonality_mode)
                            df_train = pd.DataFrame({'ds': train_series.index, 'y': train_series.values})
                            m.fit(df_train)
                            fut = m.make_future_dataframe(periods=h, freq=user_freq)
                            p = m.predict(fut)
                            return p.tail(h)['yhat']

                        def sarimax_predict(train_series, h):
                            order = (1,1,1)
                            seasonal_order = (1,1,1,12) if user_freq == 'M' else (1,1,1,7)
                            mod = sm.tsa.statespace.SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                            res = mod.fit(disp=False)
                            pred = res.get_forecast(steps=h).predicted_mean
                            return pd.Series(pred.values, index=pd.date_range(start=train_series.index[-1] + pd.tseries.frequencies.to_offset(user_freq), periods=h, freq=user_freq))

                        def hw_predict(train_series, h):
                            sp = 12 if user_freq == 'M' else 7
                            m = ExponentialSmoothing(train_series, trend='add', seasonal='add', seasonal_periods=sp)
                            r = m.fit(optimized=True)
                            p = r.forecast(steps=h)
                            return pd.Series(p.values, index=pd.date_range(start=train_series.index[-1] + pd.tseries.frequencies.to_offset(user_freq), periods=h, freq=user_freq))

                        def gb_predict(train_series, h):
                            return pd.Series(np.repeat(train_series.iloc[-1], h), index=pd.date_range(start=train_series.index[-1] + pd.tseries.frequencies.to_offset(user_freq), periods=h, freq=user_freq))

                        if model_choice == 'Prophet':
                            metrics = compute_holdout_metrics(series, 'Prophet', prophet_predict, user_freq)
                        elif model_choice == 'SARIMAX':
                            metrics = compute_holdout_metrics(series, 'SARIMAX', sarimax_predict, user_freq)
                        elif model_choice == 'Holt-Winters':
                            metrics = compute_holdout_metrics(series, 'Holt-Winters', hw_predict, user_freq)
                        else:
                            metrics = compute_holdout_metrics(series, 'GradientBoosting', gb_predict, user_freq)

                        if metrics is None:
                            st.info('Not enough data to compute holdout metrics.')
                        elif 'error' in metrics:
                            st.warning(f"Could not compute metrics: {metrics.get('error')}")
                        else:
                            st.metric('MAE', f"{metrics['MAE']:.3f}")
                            st.metric('RMSE', f"{metrics['RMSE']:.3f}")
                            st.metric('MAPE', f"{metrics['MAPE']:.2f}%")
                    except Exception as e:
                        st.exception(e)

                except Exception as e:
                    st.exception(e)

    # Anomaly Tab
    with tabs[2]:
        st.header("Anomaly Detection & Driver Analysis")
        st.markdown("Detect anomalies in the target series and check which categorical variables are associated with anomalies. Hover anomaly points for context and narrative.")
        with st.expander("Anomaly detection settings"):
            contamination = st.slider("Contamination (expected fraction of anomalies)", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
            categorical_cols = st.multiselect("Select categorical columns for driver exploration (if available)", options=[c for c in df.columns.tolist() if c not in [date_col, value_col]])
            run_anom = st.button("Run Anomaly Detection")
        if run_anom:
            try:
                series = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                context = working_df.set_index(date_col).reindex(series.index)
                mask, scores = detect_anomalies(series, contamination=contamination)
                anom_df = pd.DataFrame({'ds': series.index, 'y': series.values, 'is_anomaly': mask.values, 'score': scores.values})
                merged = anom_df.set_index('ds').join(context[categorical_cols], how='left') if len(categorical_cols) > 0 else anom_df.set_index('ds').copy()
                explanations = explain_anomalies(context, mask, categorical_cols)
                explanations_map = {}
                for ex in explanations:
                    col = ex['col']
                    for cat, frac in ex['top_anom'].items():
                        key = f"{col}||{cat}"
                        delta = frac - ex['top_base'].get(cat, 0)
                        explanations_map[key] = f"{col}='{cat}' frequency delta vs baseline: {delta*100:.1f}%"

                merged = merged.reset_index()
                merged['tooltip'] = merged.apply(lambda r: create_anomaly_tooltip(r, categorical_cols, explanations_map), axis=1)
                merged['marker_color'] = np.where(merged['is_anomaly'], 'red', 'blue')
                merged['marker_symbol'] = np.where(merged['is_anomaly'], 'x', 'circle')

                st.write(f"Detected {merged['is_anomaly'].sum()} anomalies out of {len(merged)} points")

                st.subheader('Narrative Summary')
                if len(explanations) == 0:
                    st.write('No strong categorical drivers detected (or none selected).')
                else:
                    for ex in explanations:
                        st.markdown(f"**{ex['col']}** — Top categories during anomalies:")
                        for k, v in ex['top_anom'].items():
                            st.write(f"- {k}: {v*100:.1f}% during anomalies")
                        for n in ex['narrative']:
                            st.caption(n)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=merged['ds'], y=merged['y'], mode='lines+markers', name='value', marker=dict(color=merged['marker_color'], symbol=merged['marker_symbol'], size=8), hoverinfo='text', hovertext=merged['tooltip']))
                anoms = merged[merged['is_anomaly'] == True]
                if not anoms.empty:
                    fig.add_trace(go.Scatter(x=anoms['ds'], y=anoms['y'], mode='markers+text', name='anomalies', marker=dict(color='red', size=12, symbol='x'), text=['Anomaly'] * len(anoms), textposition='top center', hoverinfo='text', hovertext=anoms['tooltip']))
                fig.update_layout(title='Series with anomalies highlighted', xaxis_title='Date', yaxis_title=value_col)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown('---')
                st.subheader('Anomaly table (top rows)')
                st.dataframe(merged.sort_values('score').head(50))
                download_df(merged.reset_index(), 'anomalies_with_context.csv')

            except Exception as e:
                st.exception(e)

    # Model Compare Tab
    with tabs[3]:
        st.header("Model Compare")
        st.markdown("Run multiple models and compare forecast outputs and holdout metrics side-by-side.")
        with st.expander("Comparison settings"):
            compare_models = st.multiselect("Select models to compare", ['Prophet', 'SARIMAX', 'Holt-Winters', 'GradientBoosting'], default=['Prophet', 'SARIMAX'])
            comp_periods = st.number_input("Horizon for comparison", min_value=1, max_value=365, value=12)
            run_compare = st.button("Run Comparison")
        if run_compare:
            results = {}
            metrics = []
            try:
                for mc in compare_models:
                    try:
                        if mc == 'Prophet' and Prophet is not None:
                            m, fcst = forecast_with_prophet(working_df, date_col, value_col, comp_periods, user_freq, 'additive', True, False, False)
                            if 'yhat' in fcst.columns:
                                out = fcst[['ds', 'yhat']].rename(columns={'yhat': 'y'})
                            elif 'y' in fcst.columns:
                                out = fcst[['ds', 'y']]
                            else:
                                out = fcst[['ds']].copy()
                            results[mc] = out
                            def p_predict(train, h):
                                m = Prophet(yearly_seasonality=True, weekly_seasonality=False)
                                m.fit(pd.DataFrame({'ds': train.index, 'y': train.values}))
                                fut = m.make_future_dataframe(periods=h, freq=user_freq)
                                return m.predict(fut).tail(h)['yhat']
                            met = compute_holdout_metrics(working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill'), mc, p_predict, user_freq)
                            if met:
                                metrics.append(met)

                        elif mc == 'SARIMAX':
                            res, fcst = sarimax_forecast(working_df, date_col, value_col, comp_periods, user_freq)
                            out = fcst.rename(columns={'yhat': 'y'})[['ds', 'y']]
                            results[mc] = out
                            def s_predict(train, h):
                                order = (1, 1, 1)
                                seasonal_order = (1, 1, 1, 12) if user_freq == 'M' else (1, 1, 1, 7)
                                mod = sm.tsa.statespace.SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
                                r = mod.fit(disp=False)
                                return r.get_forecast(steps=h).predicted_mean
                            met = compute_holdout_metrics(working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill'), mc, s_predict, user_freq)
                            if met:
                                metrics.append(met)

                        elif mc == 'Holt-Winters':
                            res, fcst = holt_winters_forecast(working_df, date_col, value_col, comp_periods, user_freq)
                            out = fcst.rename(columns={'yhat': 'y'})[['ds', 'y']]
                            results[mc] = out
                            def h_predict(train, h):
                                sp = 12 if user_freq == 'M' else 7
                                m = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=sp)
                                r = m.fit(optimized=True)
                                return r.forecast(steps=h)
                            met = compute_holdout_metrics(working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill'), mc, h_predict, user_freq)
                            if met:
                                metrics.append(met)

                        elif mc == 'GradientBoosting':
                            model, fcst = ml_forecast_with_gb(working_df, date_col, value_col, comp_periods, user_freq)
                            out = fcst.rename(columns={'yhat': 'y'})[['ds', 'y']]
                            results[mc] = out
                            def g_predict(train, h):
                                return pd.Series(np.repeat(train.iloc[-1], h), index=pd.date_range(start=train.index[-1] + pd.tseries.frequencies.to_offset(user_freq), periods=h, freq=user_freq))
                            met = compute_holdout_metrics(working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill'), mc, g_predict, user_freq)
                            if met:
                                metrics.append(met)

                    except Exception as e:
                        st.warning(f"Model {mc} failed: {e}")

                merged = None
                for k, v in results.items():
                    v2 = v.set_index('ds')
                    v2.columns = [k]
                    if merged is None:
                        merged = v2
                    else:
                        merged = merged.join(v2, how='outer')
                if merged is not None:
                    st.subheader('Forecast comparison chart')
                    st.line_chart(merged)
                    combined = merged.reset_index()
                    download_df(combined, 'model_comparison_forecasts.csv')

                if len(metrics) > 0:
                    st.subheader('Comparison metrics (holdout)')
                    mdf = metrics
                    # HTML pill table
                    html = metrics_to_html_table(mdf)
                    st.markdown(html, unsafe_allow_html=True)
                    download_df(pd.DataFrame(mdf), 'model_comparison_metrics.csv')
                else:
                    st.info('No metrics available — not enough data or models failed.')

            except Exception as e:
                st.exception(e)

    # Summary & Notes Tab
    with tabs[4]:
        st.header("Summary & Notes")
        st.markdown("This section automatically highlights the best model (lowest MAPE) and explains why we recommend it.")
        try:
            # metrics variable may exist from previous comparison run
            if 'metrics' in locals() and isinstance(metrics, list) and len(metrics) > 0:
                best = pick_best_model(metrics)
                if best is None:
                    st.write('Could not determine best model (no valid MAPE).')
                else:
                    st.subheader('Recommended Model')
                    st.markdown(f"**{best.get('model')}** — chosen because it has the lowest MAPE ({best.get('MAPE'):.2f}%) among compared models.")
                    st.markdown(interpret_metrics(best))
            else:
                st.info('Run Model Compare first to populate metrics and get a recommendation.')
        except Exception as e:
            st.exception(e)

    st.markdown("---")
    st.caption(FOOTER)


if __name__ == '__main__':
    main()
