import warnings
warnings.filterwarnings("ignore")

try:
    # Try the new ARIMA API (statsmodels >=0.12)
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    # Fallback to legacy ARIMA API (statsmodels <0.12)
    from statsmodels.tsa.arima_model import ARIMA
    print("Warning: Using legacy ARIMA API. Consider upgrading statsmodels>=0.12 for full compatibility.")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from concurrent.futures import ThreadPoolExecutor
import warnings
import io
import json
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>
body {background-color: #f5f5f5;}
h1, h2, h3 {color: #1F2937;}
.stButton>button {background-color:#4F46E5; color:white;}
.stSidebar .sidebar-content {background-color: #E0E7FF;}
.metric-card {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Animated loader HTML/CSS
# -------------------------------
loader_html = """
<div style="display:flex;justify-content:center;align-items:center;height:50px;">
  <div class="lds-ring"><div></div><div></div><div></div><div></div></div>
</div>
<style>
.lds-ring {display: inline-block; position: relative; width: 40px; height: 40px;}
.lds-ring div {box-sizing: border-box; display: block; position: absolute; width: 32px; height: 32px; margin: 4px; border: 4px solid #4F46E5; border-radius: 50%; animation: lds-ring 1.2s cubic-bezier(0.5, 0, 0.5, 1) infinite; border-color: #4F46E5 transparent transparent transparent;}
.lds-ring div:nth-child(1) { animation-delay: -0.45s; }
.lds-ring div:nth-child(2) { animation-delay: -0.3s; }
.lds-ring div:nth-child(3) { animation-delay: -0.15s; }
@keyframes lds-ring {0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); }}
</style>
"""

# -------------------------------
# Utility Functions
# -------------------------------
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

def detect_time_column(df):
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            return col
        except:
            continue
    return None

def preprocess_data(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(time_col)
    df = df.dropna(subset=[time_col])
    return df

# -------------------------------
# Forecasting functions with caching
# -------------------------------
@st.cache_data(show_spinner=False)
def cached_forecast_arima(series, steps):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    return model_fit.forecast(steps=steps)

@st.cache_data(show_spinner=False)
def cached_forecast_ets(series, steps):
    model = ExponentialSmoothing(series, seasonal=None)
    model_fit = model.fit()
    return model_fit.forecast(steps)

@st.cache_data(show_spinner=False)
def cached_forecast_prophet(df, time_col, value_col, periods, holidays=None, custom_seasonality=None):
    df_prophet = df[[time_col, value_col]].rename(columns={time_col:'ds', value_col:'y'})
    model = Prophet(holidays=holidays)
    if custom_seasonality:
        for season in custom_seasonality:
            model.add_seasonality(name=season['name'], period=season['period'], fourier_order=season['fourier_order'])
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']]

# -------------------------------
# Anomaly detection
# -------------------------------
def detect_anomalies_stat(series):
    mean = series.mean()
    std = series.std()
    z_score = (series - mean)/std
    return series[np.abs(z_score) > 3]

def detect_anomalies_iforest(series):
    clf = IsolationForest(contamination=0.05, random_state=42)
    data = series.values.reshape(-1,1)
    clf.fit(data)
    pred = clf.predict(data)
    return series[pred==-1]

# -------------------------------
# Insights & plotting
# -------------------------------
def generate_insights(df, metric, anomalies):
    insights = []
    series = df[metric]
    if series.iloc[-1] > series.mean():
        insights.append(f"Metric {metric} is currently above its historical average.")
    if len(anomalies)>0:
        for date, value in anomalies.items():
            insights.append(f"Anomaly detected in {metric} on {date.date()} with value {value:.2f}.")
    return insights

def compute_model_metrics(series, forecast):
    min_len = min(len(series), len(forecast))
    rmse = mean_squared_error(series[-min_len:], forecast[-min_len:], squared=False)
    mape = mean_absolute_percentage_error(series[-min_len:], forecast[-min_len:])
    return rmse, mape

def plot_forecast(df, forecast_df, time_col, metric):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[metric], mode='lines', name='Actual'))
    if 'yhat' in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], mode='lines', name='Upper', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], mode='lines', name='Lower', line=dict(dash='dash')))
    else:
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df, mode='lines', name='Forecast'))
    fig.update_layout(title=f"Forecast for {metric}", xaxis_title=time_col, yaxis_title=metric)
    return fig

def plot_anomalies(df, time_col, metric, anomalies):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[metric], mode='lines', name='Actual'))
    if len(anomalies)>0:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies.values, mode='markers', name='Anomalies', marker=dict(color='red', size=10)))
    fig.update_layout(title=f"Anomalies in {metric}", xaxis_title=time_col, yaxis_title=metric)
    return fig

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    time_col_auto = detect_time_column(df)
    time_col = st.sidebar.selectbox("Select Time Column", df.columns, index=df.columns.get_loc(time_col_auto) if time_col_auto else 0)
    df = preprocess_data(df, time_col)

    metric_options = df.select_dtypes(include=np.number).columns.tolist()
    metrics_selected = st.sidebar.multiselect("Select Metrics to Forecast", metric_options, default=metric_options[:1])

    # ------------------ Interactive Horizon ------------------
    horizon_dict = {}
    st.sidebar.markdown("**Forecast Horizon per Metric (days)**")
    for metric in metrics_selected:
        horizon_dict[metric] = st.sidebar.slider(f"{metric} horizon", min_value=1, max_value=365, value=7)

    models_selected = st.sidebar.multiselect("Select Models", ["ARIMA","ETS","Prophet"], default=["ARIMA","Prophet"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Optional:** Upload holidays CSV with column 'ds' for Prophet")
    holiday_file = st.sidebar.file_uploader("Upload Holidays CSV", type=["csv"])
    holidays_df = None
    if holiday_file:
        holidays_df = pd.read_csv(holiday_file)
        if 'ds' in holidays_df.columns:
            holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
        else:
            st.sidebar.warning("Holiday file must have a 'ds' column")

    st.sidebar.markdown("**Optional:** Add custom seasonalities for Prophet")
    custom_seasonality_input = st.sidebar.text_area("Enter as JSON list e.g., [{'name':'monthly','period':30.5,'fourier_order':5}]", "")
    custom_seasonality = None
    if custom_seasonality_input:
        try:
            custom_seasonality = json.loads(custom_seasonality_input)
        except:
            st.sidebar.warning("Invalid JSON format for custom seasonalities")

# -------------------------------
# Forecasting function for threading
# -------------------------------
def run_forecast_for_metric(metric):
    horizon = horizon_dict[metric]
    results = {}
    if "ARIMA" in models_selected:
        try:
            arima_forecast = cached_forecast_arima(df[metric], horizon)
            rmse, mape = compute_model_metrics(df[metric], arima_forecast)
            results['ARIMA'] = {'forecast': arima_forecast, 'rmse': rmse, 'mape': mape}
        except:
            st.warning(f"ARIMA failed for {metric}")
    if "ETS" in models_selected:
        try:
            ets_forecast = cached_forecast_ets(df[metric], horizon)
            rmse, mape = compute_model_metrics(df[metric], ets_forecast)
            results['ETS'] = {'forecast': ets_forecast, 'rmse': rmse, 'mape': mape}
        except:
            st.warning(f"ETS failed for {metric}")
    if "Prophet" in models_selected:
        try:
            prophet_forecast = cached_forecast_prophet(df, time_col, metric, horizon,
                                                        holidays=holidays_df,
                                                        custom_seasonality=custom_seasonality)
            y_true = df[metric].values[-horizon:]
            y_pred = prophet_forecast['yhat'].values[-horizon:]
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            results['Prophet'] = {'forecast': prophet_forecast, 'rmse': rmse, 'mape': mape}
        except:
            st.warning(f"Prophet failed for {metric}")
    return metric, results

# -------------------------------
# Main Panel
# -------------------------------
if uploaded_file:
    st.title("Time Series Forecasting Dashboard")
    tabs = st.tabs(["Data Preview","Forecasting","Anomalies","Insights","Download Results","Model Comparison"])

    # --- Data Preview ---
    with tabs[0]:
        st.subheader("Raw Data")
        st.dataframe(df.head())
        st.write("Summary Statistics")
        st.dataframe(df.describe())

    # --- Forecasting ---
    with tabs[1]:
        st.subheader("Forecasts")
        forecast_results = {}
        loader_placeholder = st.empty()
        with loader_placeholder.container():
            st.markdown(loader_html, unsafe_allow_html=True)
            st.write("Running multi-metric forecasting. Please wait...")
            with ThreadPoolExecutor(max_workers=min(4, len(metrics_selected))) as executor:
                futures = [executor.submit(run_forecast_for_metric, metric) for metric in metrics_selected]
                for future in futures:
                    metric, results = future.result()
                    forecast_results[metric] = results
        loader_placeholder.empty()

        for metric in metrics_selected:
            st.markdown(f"### Forecast for {metric}")
            results = forecast_results[metric]
            metrics_df = pd.DataFrame({
                'Model': [k for k in results.keys()],
                'RMSE': [v['rmse'] for v in results.values()],
                'MAPE': [v['mape'] for v in results.values()]
            })
            best_model = metrics_df.sort_values('RMSE').iloc[0]['Model']
            forecast_df = results[best_model]['forecast']
            st.write(f"**Selected Model:** {best_model}")
            st.dataframe(metrics_df)
            fig = plot_forecast(df, forecast_df, time_col, metric)
            st.plotly_chart(fig, use_container_width=True)

            if 'yhat' in forecast_df.columns:
                forecast_value = forecast_df['yhat'].iloc[-1]
            else:
                forecast_value = forecast_df.iloc[-1]
            current_value = df[metric].iloc[-1]
            st.markdown(f"""
                <div class="metric-card">
                    <h4>{metric} Forecast</h4>
                    <p>Current Value: {current_value:.2f}</p>
                    <p>Forecasted Value: {forecast_value:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

    # --- Anomalies ---
    with tabs[2]:
        st.subheader("Anomaly Detection")
        anomalies_all = {}
        for metric in metrics_selected:
            st.markdown(f"### Anomalies in {metric}")
            anomalies_stat = detect_anomalies_stat(df[metric])
            anomalies_if = detect_anomalies_iforest(df[metric])
            anomalies_combined = pd.concat([anomalies_stat, anomalies_if]).drop_duplicates()
            anomalies_all[metric] = anomalies_combined
            fig_anom = plot_anomalies(df, time_col, metric, anomalies_combined)
            st.plotly_chart(fig_anom, use_container_width=True)
            st.write(f"Detected {len(anomalies_combined)} anomalies.")

    # --- Insights ---
    with tabs[3]:
        st.subheader("Auto-generated Insights")
        for metric in metrics_selected:
            anomalies_combined = anomalies_all.get(metric, pd.Series())
            insights = generate_insights(df, metric, anomalies_combined)
            st.markdown(f"### Insights for {metric}")
            for ins in insights:
                st.write("- " + ins)

    # --- Download Results ---
    with tabs[4]:
        st.subheader("Download Forecast Results")
        for metric in metrics_selected:
            results = forecast_results[metric]
            best_model = pd.DataFrame({
                'Model': [k for k in results.keys()],
                'RMSE': [v['rmse'] for v in results.values()],
                'MAPE': [v['mape'] for v in results.values()]
            }).sort_values('RMSE').iloc[0]['Model']
            forecast_df = results[best_model]['forecast']
            st.markdown(f"**{metric} - {best_model} Forecast**")
            csv = forecast_df.to_csv(index=False).encode()
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                forecast_df.to_excel(writer, sheet_name=f'{metric}_Forecast', index=False)
            st.download_button(label=f"Download {metric} CSV", data=csv, file_name=f"{metric}_forecast.csv", mime='text/csv')
            st.download_button(label=f"Download {metric} Excel", data=excel_buffer, file_name=f"{metric}_forecast.xlsx", mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    # --- Model Comparison ---
    with tabs[5]:
        st.subheader("Model Comparison (RMSE / MAPE)")
        comparison_rows = []
        for metric in metrics_selected:
            results = forecast_results[metric]
            for model_name, metrics in results.items():
                comparison_rows.append({'Metric': metric, 'Model': model_name, 'RMSE': metrics['rmse'], 'MAPE': metrics['mape']})
        comparison_df = pd.DataFrame(comparison_rows)
        fig_rmse = px.bar(comparison_df, x='Metric', y='RMSE', color='Model', barmode='group', title="RMSE Comparison")
        fig_mape = px.bar(comparison_df, x='Metric', y='MAPE', color='Model', barmode='group', title="MAPE Comparison")
        st.plotly_chart(fig_rmse, use_container_width=True)
        st.plotly_chart(fig_mape, use_container_width=True)
