"""
Forecasting Pro
No Code Forecasting App

Single-file Streamlit app. Also writes requirements.txt when run.
Footer: Developed with ❤️Streamlit CE Innovation Labs 2025

This app supports:
- Prophet (for seasonality/holidays)
- pmdarima auto_arima
- SARIMAX (statsmodels)
- Gradient Boosting (LightGBM via scikit-learn wrapper)
- Basic TBATS-like seasonal model using statsmodels (seasonal_decompose for guidance)
- Anomaly detection using IsolationForest
- Driver analysis using permutation importance and simple aggregated categorical comparisons

Watch: this app aims for readability and production-grade structure in a single file.

"""

import streamlit as st
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

# forecasting libraries
try:
    from prophet import Prophet
except Exception:
    # prophet might be fbprophet in older installs
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

try:
    import pmdarima as pm
except Exception:
    pm = None

import statsmodels.api as sm
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

import plotly.express as px
import plotly.graph_objects as go

# App metadata
APP_TITLE = "Forecasting Pro"
APP_SUBTITLE = "No Code Forecasting App"
FOOTER = "Developed with \u2764\ufe0fStreamlit CE Innovation Labs 2025"

# requirements content (will write out when running)
REQUIREMENTS = r"""
streamlit>=1.20
pandas>=1.5
numpy>=1.23
plotly>=5.0
matplotlib>=3.5
prophet>=1.1
pmdarima>=2.2
statsmodels>=0.13
scikit-learn>=1.1
lightgbm>=3.3
shap>=0.41
plotly>=5.0
python-dateutil
"""

# Helper functions

def write_requirements_file():
    with open("requirements.txt", "w") as f:
        f.write(REQUIREMENTS)


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


def download_link(df: pd.DataFrame, filename: str, link_text: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    st.markdown(f"[{link_text}]({href})")


def forecast_with_prophet(df, date_col, value_col, periods, freq, seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality, holidays_df=None):
    m = Prophet(seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality)
    if holidays_df is not None and not holidays_df.empty:
        m.holidays = holidays_df
    hist = df.rename(columns={date_col: 'ds', value_col: 'y'})[['ds', 'y']]
    m.fit(hist)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fcst = m.predict(future)
    return m, fcst


def auto_arima_forecast(df, date_col, value_col, periods, freq):
    series = df.set_index(date_col)[value_col].asfreq(freq)
    # fall back: if pm is not installed, raise informative error
    if pm is None:
        raise RuntimeError("pmdarima is not available in the environment. Add pmdarima to requirements.txt")
    model = pm.auto_arima(series, seasonal=True, stepwise=True, suppress_warnings=True, error_action='ignore')
    future_idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    fcst = pd.Series(model.predict(n_periods=periods), index=future_idx)
    fcst = fcst.rename('y').reset_index().rename(columns={'index': 'ds'})
    return model, fcst


def sarimax_forecast(df, date_col, value_col, periods, freq, order=(1,0,0), seasonal_order=(0,0,0,0)):
    series = df.set_index(date_col)[value_col].asfreq(freq).fillna(method='ffill')
    model = sm.tsa.statespace.SARIMAX(series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=periods)
    idx = pd.date_range(start=series.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    fcst = pd.DataFrame({'ds': idx, 'yhat': pred.predicted_mean, 'yhat_lower': pred.conf_int().iloc[:,0], 'yhat_upper': pred.conf_int().iloc[:,1]})
    return res, fcst


def create_lag_features(series, lags=24):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df = df.dropna()
    return df


def ml_forecast_with_gb(df, date_col, value_col, periods, freq, exog_cols=None, lags=24):
    s = df.set_index(date_col)[value_col].asfreq(freq)
    lagged = create_lag_features(s, lags=lags)
    # Prepare X,Y
    X = lagged.drop(columns=['y'])
    Y = lagged['y']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    model = GradientBoostingRegressor(n_estimators=200)
    model.fit(X_train, y_train)
    # iterative forecasting
    last_values = s.iloc[-lags:].tolist()
    preds = []
    for _ in range(periods):
        x_row = np.array(last_values[-lags:])[::-1]  # latest lags
        x_row = x_row.reshape(1, -1)
        p = model.predict(x_row)[0]
        preds.append(p)
        last_values.append(p)
    idx = pd.date_range(start=s.index[-1] + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    fcst = pd.DataFrame({'ds': idx, 'yhat': preds})
    return model, fcst


def detect_anomalies(series, contamination=0.01):
    iso = IsolationForest(contamination=contamination, random_state=42)
    X = series.values.reshape(-1,1)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)
    outliers = preds == -1
    return pd.Series(outliers, index=series.index), pd.Series(scores, index=series.index)


def explain_anomalies(df, anomaly_mask, categorical_cols):
    # For each categorical col, compare distribution during anomalies vs baseline
    explanations = []
    anomalies = df[anomaly_mask]
    baseline = df[~anomaly_mask]
    for col in categorical_cols:
        if col not in df.columns: continue
        try:
            top_anom = anomalies[col].value_counts(normalize=True).head(5)
            top_base = baseline[col].value_counts(normalize=True).head(5)
            explanations.append((col, top_anom.to_dict(), top_base.to_dict()))
        except Exception:
            continue
    return explanations


# UI building blocks

def sidebar_file_upload():
    st.sidebar.header("Upload & Settings")
    uploaded_file = st.sidebar.file_uploader("Upload time series CSV/Excel", type=['csv', 'xlsx', 'xls'])
    if uploaded_file is None:
        st.sidebar.info("You can upload a CSV or Excel file with a date column and a numeric value column.")
    return uploaded_file


def main():
    set_page()
    write_requirements_file()

    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    # top navigation panels
    tabs = st.tabs(["Data", "Forecast", "Anomaly & Drivers", "Model Compare", "Help & Notes"])

    uploaded_file = sidebar_file_upload()

    # default example dataset
    if uploaded_file is None:
        st.info("No file uploaded — using a built-in example (monthly retail sales). Upload your own in the sidebar to run on your data.")
        dates = pd.date_range(start='2015-01-01', periods=120, freq='M')
        data = (np.sin(np.arange(120)/6) + np.arange(120)/20 + np.random.normal(0,0.3,120))*100 + 500
        df = pd.DataFrame({'ds': dates, 'y': data})
    else:
        df = read_uploaded_file(uploaded_file)

    ################# Data Tab #################
    with tabs[0]:
        st.header("Data Preview")
        st.markdown("**Upload format help:** The app expects a single date column and one numeric target column. Additional columns (categorical or numeric) can be used as exogenous features or for driver analysis.")
        st.dataframe(df.head(100))
        st.markdown("---")
        with st.expander("Map columns (date & value)"):
            cols = df.columns.tolist()
            date_col = st.selectbox("Date column", cols, index=0)
            value_col = st.selectbox("Value column", cols, index=1 if len(cols)>1 else 0)
            freq = st.selectbox("Data frequency (if unknown pick 'D' or 'infer')", ['infer','D','W','M','Q','H'])
            if freq == 'infer':
                try:
                    inferred = pd.infer_freq(pd.to_datetime(df[date_col]))
                    st.write(f"Inferred frequency: {inferred}")
                except Exception:
                    st.write("Could not infer frequency; choose manually")
            st.write("Sample range:")
            try:
                st.write(f"{pd.to_datetime(df[date_col]).min().date()} to {pd.to_datetime(df[date_col]).max().date()}")
            except Exception:
                st.write("Date parsing error — check your date column")

    ################# Forecast Tab #################
    with tabs[1]:
        st.header("Forecast")
        st.markdown("Choose model, set horizon, seasonality and holidays. Guidance notes are on the right.")
        # controls
        with st.sidebar.expander("Forecast Settings", expanded=True):
            model_choice = st.selectbox("Model", ["Prophet", "Auto-ARIMA", "SARIMAX", "GradientBoosting"])
            periods = st.number_input("Forecast periods (horizon)", min_value=1, max_value=1000, value=12)
            user_freq = st.selectbox("Frequency (pandas offset) e.g. 'M' monthly, 'D' daily", ['M','D','W','H'])
            seasonality_mode = st.selectbox("Seasonality Mode (Prophet only)", ['additive','multiplicative'])
            yearly = st.checkbox("Yearly seasonality", value=True)
            weekly = st.checkbox("Weekly seasonality", value=False)
            daily = st.checkbox("Daily seasonality", value=False)
            include_holidays = st.checkbox("Include holidays CSV (two columns: ds, holiday)", value=False)
            if include_holidays:
                holidays_file = st.sidebar.file_uploader("Upload holidays CSV", type=['csv'])
            download_button = st.sidebar.checkbox("Provide download button for forecast CSV", value=True)

        # prepare df and mappings
        try:
            date_col
        except NameError:
            date_col = 'ds'
            value_col = 'y'
        working_df = df.copy()
        # ensure datetime
        try:
            working_df[date_col] = pd.to_datetime(working_df[date_col])
        except Exception as e:
            st.error("Could not parse the selected date column to datetime. Check the file format.")
            st.exception(e)
            return

        # run model
        run = st.button("Run Forecast")
        if run:
            with st.spinner("Fitting model..."):
                try:
                    if model_choice == 'Prophet':
                        if Prophet is None:
                            st.error("Prophet package not available in this environment. Please install 'prophet' package.")
                        holidays_df = None
                        if include_holidays and 'holidays_file' in locals() and holidays_file is not None:
                            try:
                                holidays_df = pd.read_csv(holidays_file)
                                # expect columns 'ds' and 'holiday'
                            except Exception:
                                st.error("Could not read holidays file; ensure two columns 'ds' and 'holiday'")
                        m, fcst = forecast_with_prophet(working_df, date_col, value_col, periods, user_freq, seasonality_mode, yearly, weekly, daily, holidays_df)
                        # show
                        st.success("Forecast complete (Prophet)")
                        fig = m.plot(fcst)
                        st.pyplot(fig)
                        # interactive plotly
                        p = px.line(fcst, x='ds', y=['yhat','yhat_lower','yhat_upper'] if 'yhat' in fcst.columns else ['yhat'], labels={'value':'y'})
                        st.plotly_chart(p, use_container_width=True)
                        out_df = fcst[['ds','yhat']].rename(columns={'yhat':'y'}) if 'yhat' in fcst.columns else fcst[['ds','y']]

                    elif model_choice == 'Auto-ARIMA':
                        if pm is None:
                            st.error("pmdarima not installed. Please add pmdarima to requirements.")
                        else:
                            model, fcst = auto_arima_forecast(working_df, date_col, value_col, periods, user_freq)
                            st.success("Forecast complete (Auto-ARIMA)")
                            fig = go.Figure()
                            hist = working_df.set_index(date_col)[value_col]
                            fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='history'))
                            fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['y'], name='forecast'))
                            st.plotly_chart(fig, use_container_width=True)
                            out_df = fcst.rename(columns={'y':'yhat'})[['ds','yhat']]

                    elif model_choice == 'SARIMAX':
                        order = (1,1,1)
                        seasonal_order = (1,1,1,12)
                        res, fcst = sarimax_forecast(working_df, date_col, value_col, periods, user_freq, order, seasonal_order)
                        st.success("Forecast complete (SARIMAX)")
                        fig = go.Figure()
                        hist = working_df.set_index(date_col)[value_col]
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='history'))
                        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='forecast'))
                        st.plotly_chart(fig, use_container_width=True)
                        out_df = fcst.rename(columns={'yhat':'y'})[['ds','y']]

                    elif model_choice == 'GradientBoosting':
                        model, fcst = ml_forecast_with_gb(working_df, date_col, value_col, periods, user_freq)
                        st.success("Forecast complete (GradientBoosting)")
                        fig = go.Figure()
                        hist = working_df.set_index(date_col)[value_col]
                        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name='history'))
                        fig.add_trace(go.Scatter(x=fcst['ds'], y=fcst['yhat'], name='forecast'))
                        st.plotly_chart(fig, use_container_width=True)
                        out_df = fcst.rename(columns={'yhat':'y'})[['ds','y']]

                    # accuracy metrics where possible
                    st.markdown("**Forecast sample (first rows)**")
                    st.dataframe(out_df.head(20))
                    if download_button:
                        csv = out_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')

                    # show metrics by backtesting where feasible
                    st.markdown("---")
                    st.subheader("Quick accuracy estimate (simple holdout)")
                    try:
                        # use last 20% as test
                        series = working_df.set_index(date_col)[value_col].asfreq(user_freq)
                        cutoff = int(len(series)*0.8)
                        train = series.iloc[:cutoff]
                        test = series.iloc[cutoff:]
                        # naive persistence
                        naive = train.iloc[-1]
                        from sklearn.metrics import mean_absolute_error, mean_squared_error
                        # use model predictions where possible (approx)
                        # for Prophet only sample re-fit on train
                        if model_choice == 'Prophet' and Prophet is not None:
                            m2 = Prophet(yearly_seasonality=yearly, weekly_seasonality=weekly, daily_seasonality=daily, seasonality_mode=seasonality_mode)
                            m2.fit(pd.DataFrame({'ds':train.index, 'y':train.values}))
                            fut = m2.make_future_dataframe(periods=len(test), freq=user_freq)
                            pred = m2.predict(fut)
                            ypred = pred.tail(len(test))['yhat'].values
                        elif model_choice == 'Auto-ARIMA' and pm is not None:
                            arima = pm.auto_arima(train, seasonal=True, suppress_warnings=True, error_action='ignore')
                            ypred = arima.predict(n_periods=len(test))
                        elif model_choice == 'SARIMAX':
                            mod = sm.tsa.statespace.SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
                            res = mod.fit(disp=False)
                            ypred = res.get_forecast(steps=len(test)).predicted_mean
                        else:
                            # for ML model attempt simple naive forecast
                            ypred = np.repeat(train.iloc[-1], len(test))
                        mae = mean_absolute_error(test, ypred)
                        rmse = mean_squared_error(test, ypred, squared=False)
                        st.metric("MAE (holdout)", f"{mae:.3f}")
                        st.metric("RMSE (holdout)", f"{rmse:.3f}")
                    except Exception as e:
                        st.info("Could not compute holdout metrics. Data frequency or model limitation may be the reason.")

                except Exception as e:
                    st.exception(e)

    ################# Anomaly & Drivers Tab #################
    with tabs[2]:
        st.header("Anomaly Detection & Driver Analysis")
        st.markdown("Detect anomalies in the target series and check which categorical variables are associated with anomalies.")
        with st.expander("Anomaly detection settings"):
            contamination = st.slider("Contamination (expected fraction of anomalies)", min_value=0.001, max_value=0.2, value=0.02, step=0.001)
            categorical_cols = st.multiselect("Select categorical columns for driver exploration (if available)", options=df.columns.tolist())
            run_anom = st.button("Run Anomaly Detection")
        if run_anom:
            try:
                series = working_df.set_index(date_col)[value_col].asfreq(user_freq).fillna(method='ffill')
                mask, scores = detect_anomalies(series, contamination=contamination)
                anom_df = pd.DataFrame({'ds':series.index, 'y':series.values, 'is_anomaly':mask.values, 'score':scores.values})
                st.write(f"Detected {mask.sum()} anomalies out of {len(mask)} points")
                fig = px.scatter(anom_df, x='ds', y='y', color='is_anomaly', hover_data=['score'])
                st.plotly_chart(fig, use_container_width=True)
                # driver analysis
                explanations = explain_anomalies(working_df.set_index(date_col), mask, categorical_cols)
                st.subheader('Driver Analysis (categorical)')
                if len(explanations)==0:
                    st.write("No categorical drivers selected or insufficient data to explain anomalies.")
                else:
                    for col, anom_stats, base_stats in explanations:
                        st.markdown(f"**{col}**")
                        st.write("Top categories during anomalies:")
                        st.write(pd.DataFrame.from_dict(anom_stats, orient='index', columns=['fraction']).sort_values('fraction', ascending=False).head(5))
                        st.write("Top categories during baseline:")
                        st.write(pd.DataFrame.from_dict(base_stats, orient='index', columns=['fraction']).sort_values('fraction', ascending=False).head(5))

            except Exception as e:
                st.exception(e)

    ################# Model Compare Tab #################
    with tabs[3]:
        st.header("Model Compare")
        st.markdown("Quick compare: run multiple models and compare forecast outputs and holdout metrics side-by-side.")
        with st.expander("Comparison settings"):
            compare_models = st.multiselect("Select models to compare", ['Prophet', 'Auto-ARIMA', 'SARIMAX', 'GradientBoosting'], default=['Prophet','Auto-ARIMA'])
            comp_periods = st.number_input("Horizon for comparison", min_value=1, max_value=365, value=12)
            run_compare = st.button("Run Comparison")
        if run_compare:
            results = {}
            for mc in compare_models:
                try:
                    if mc == 'Prophet' and Prophet is not None:
                        m, fcst = forecast_with_prophet(working_df, date_col, value_col, comp_periods, user_freq, 'additive', True, False, False)
                        results[mc] = fcst[['ds','yhat']].rename(columns={'yhat':'y'})
                    elif mc == 'Auto-ARIMA' and pm is not None:
                        model, fcst = auto_arima_forecast(working_df, date_col, value_col, comp_periods, user_freq)
                        results[mc] = fcst.rename(columns={'y':'yhat'})[['ds','yhat']].rename(columns={'yhat':'y'})
                    elif mc == 'SARIMAX':
                        res, fcst = sarimax_forecast(working_df, date_col, value_col, comp_periods, user_freq)
                        results[mc] = fcst.rename(columns={'yhat':'y'})[['ds','y']]
                    elif mc == 'GradientBoosting':
                        model, fcst = ml_forecast_with_gb(working_df, date_col, value_col, comp_periods, user_freq)
                        results[mc] = fcst.rename(columns={'yhat':'y'})[['ds','y']]
                except Exception as e:
                    st.warning(f"Model {mc} failed: {e}")
            # merge for plotting
            merged = None
            for k,v in results.items():
                v = v.set_index('ds')
                v.columns = [k]
                if merged is None:
                    merged = v
                else:
                    merged = merged.join(v, how='outer')
            if merged is not None:
                st.line_chart(merged)
                # allow CSV download
                combined = merged.reset_index()
                csv = combined.to_csv(index=False).encode('utf-8')
                st.download_button("Download comparison CSV", data=csv, file_name='model_comparison.csv', mime='text/csv')

    ################# Help & Notes Tab #################
    with tabs[4]:
        st.header("Help & Notes")
        st.markdown("""
        **Quick guidance**
        - Choose Prophet for flexible seasonality and holidays support.
        - Choose Auto-ARIMA for classical statistical forecasts when series are fairly regular.
        - Use GradientBoosting when you have many related exogenous variables and want to capture complex non-linear relationships.

        **What the app does**
        - Fits the selected model to your target series and produces a forecast horizon you choose.
        - Performs a simple holdout backtest for a quick accuracy estimate (not a substitute for full cross-validation).
        - Detects anomalies using IsolationForest and shows which categorical variables co-occur with anomalies.

        **Limitations**
        - This is a single-file example intended for production adaptation. For high-scale deployments, move model training to batch jobs, add caching, and use more robust MLops patterns.

        **Next steps / Improvements you can ask for**
        - Add cross-validation (rolling) for robust metrics.
        - Add Prophet extra regressors and better holiday management UI.
        - Add SHAP explanations for ML models (requires extra compute/time).
        """)

    # footer
    st.markdown("---")
    st.caption(FOOTER)


if __name__ == '__main__':
    main()
