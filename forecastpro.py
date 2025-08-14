# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
import io
import json

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Time Series Forecasting Dashboard", layout="wide")

# -------------------------------
# Import forecasting libraries
# -------------------------------
try:
    from neuralprophet import NeuralProphet
except ImportError:
    st.error("NeuralProphet not installed properly. Check requirements.txt for PyTorch dependencies.")
import pmdarima as pm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor

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
# Loader HTML
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
# Forecasting functions
# -------------------------------
@st.cache_data(show_spinner=False)
def cached_forecast_auto_arima(series, steps):
    model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
    forecast = model.predict(n_periods=steps)
    return forecast

@st.cache_data(show_spinner=False)
def cached_forecast_neuralprophet(df, time_col, value_col, periods):
    df_np = df[[time_col, value_col]].rename(columns={time_col:'ds', value_col:'y'})
    model = NeuralProphet()
    model.fit(df_np, freq='D')
    future = model.make_future_dataframe(df_np, periods=periods)
    forecast = model.predict(future)
    return forecast[['ds','yhat1']]

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
# Plotting functions
# -------------------------------
def plot_forecast(df, forecast_df, time_col, metric):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[time_col], y=df[metric], mode='lines', name='Actual'))
    if 'yhat1' in forecast_df.columns:
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat1'], mode='lines', name='Forecast'))
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

def compute_model_metrics(series, forecast):
    min_len = min(len(series), len(forecast))
    rmse = mean_squared_error(series[-min_len:], forecast[-min_len:], squared=False)
    mape = mean_absolute_percentage_error(series[-min_len:], forecast[-min_len:])
    return rmse, mape

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)
    time_col_auto = detect_time_column(df)
    time_col = st.sidebar.selectbox("Select Time Column", df.columns, index=df.columns.get_loc(time_col_auto) if time_col_auto else 0)
    df = preprocess_data(df, time_col)
    
    metric_options = df.select_dtypes(include=np.number).columns.tolist()
    metrics_selected = st.sidebar.multiselect("Select Metrics", metric_options, default=metric_options[:1])
    
    horizon_dict = {}
    st.sidebar.markdown("**Forecast Horizon per Metric (days)**")
    for metric in metrics_selected:
        horizon_dict[metric] = st.sidebar.slider(f"{metric} horizon", 1, 365, 7)
    
    models_selected = st.sidebar.multiselect("Select Models", ["Auto-ARIMA","NeuralProphet"], default=["Auto-ARIMA","NeuralProphet"])

# -------------------------------
# Forecasting function
# -------------------------------
def run_forecast_for_metric(metric):
    horizon = horizon_dict[metric]
    results = {}
    if "Auto-ARIMA" in models_selected:
        try:
            arima_forecast = cached_forecast_auto_arima(df[metric], horizon)
            rmse, mape = compute_model_metrics(df[metric], arima_forecast)
            results['Auto-ARIMA'] = {'forecast': arima_forecast, 'rmse': rmse, 'mape': mape}
        except:
            st.warning(f"Auto-ARIMA failed for {metric}")
    if "NeuralProphet" in models_selected:
        try:
            np_forecast = cached_forecast_neuralprophet(df, time_col, metric, horizon)
            y_true = df[metric].values[-horizon:]
            y_pred = np_forecast['yhat1'].values[-horizon:]
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            results['NeuralProphet'] = {'forecast': np_forecast, 'rmse': rmse, 'mape': mape}
        except:
            st.warning(f"NeuralProphet failed for {metric}")
    return metric, results

# -------------------------------
# Main Panel
# -------------------------------
if uploaded_file:
    st.title("Time Series Forecasting Dashboard")
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
            'Model':[k for k in results.keys()],
            'RMSE':[v['rmse'] for v in results.values()],
            'MAPE':[v['mape'] for v in results.values()]
        })
        best_model = metrics_df.sort_values('RMSE').iloc[0]['Model']
        forecast_df = results[best_model]['forecast']
        st.write(f"**Selected Model:** {best_model}")
        st.dataframe(metrics_df)
        fig = plot_forecast(df, forecast_df, time_col, metric)
        st.plotly_chart(fig, use_container_width=True)
