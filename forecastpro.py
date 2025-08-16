# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor
import warnings
import io
import json
import base64
from typing import Optional, Dict, Any, List, Tuple

# Handle optional imports gracefully
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

# Import statsmodels as fallback for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings("ignore")

# Page configuration with custom theme
st.set_page_config(
    page_title="📈 AI Forecasting Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Enhanced CSS Styling
# -------------------------------
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Custom metric cards */
    .metric-card {
        background: linear-gradient(135deg, #fff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: #1e293b;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        padding: 0.7rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 15px;
        border: 2px dashed #cbd5e0;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Alert styling */
    .element-container .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loader {
        width: 50px;
        height: 50px;
        border: 4px solid #f3f4f6;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Status indicators */
    .status-success {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Model comparison cards */
    .model-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .model-winner {
        border: 2px solid #10b981;
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Enhanced Loader Component
# -------------------------------
def show_loading_animation(message: str = "Processing..."):
    return st.markdown(f"""
    <div class="loading-container">
        <div class="loader"></div>
        <p style="margin-left: 1rem; color: #64748b; font-weight: 500;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Utility Functions
# -------------------------------
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    """Detect datetime column with improved logic"""
    time_keywords = ['date', 'time', 'timestamp', 'datetime', 'created', 'updated']
    
    # First, check columns with time-related keywords
    for col in df.columns:
        if any(keyword in col.lower() for keyword in time_keywords):
            try:
                pd.to_datetime(df[col].iloc[:100])  # Test first 100 rows
                return col
            except:
                continue
    
    # Then check all columns
    for col in df.columns:
        try:
            pd.to_datetime(df[col].iloc[:100])  # Test first 100 rows
            return col
        except:
            continue
    return None

def preprocess_data(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """Enhanced data preprocessing"""
    try:
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.sort_values(time_col)
        df = df.dropna(subset=[time_col])
        
        # Remove duplicates
        df = df.drop_duplicates(subset=[time_col])
        
        # Handle missing values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return df

# -------------------------------
# Enhanced Forecasting Functions
# -------------------------------
def simple_arima_forecast(series: pd.Series, steps: int) -> np.ndarray:
    """Fallback ARIMA implementation using statsmodels"""
    try:
        if STATSMODELS_AVAILABLE:
            # Use simple ARIMA(1,1,1) as fallback
            model = ARIMA(series.dropna(), order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=steps)
            return forecast.values
        else:
            # Simple linear trend fallback
            X = np.arange(len(series)).reshape(-1, 1)
            y = series.values
            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(series), len(series) + steps).reshape(-1, 1)
            return model.predict(future_X)
    except Exception as e:
        # Ultimate fallback: last value
        st.warning(f"ARIMA failed, using last value: {str(e)}")
        return np.full(steps, series.iloc[-1])

@st.cache_data(show_spinner=False)
def cached_forecast_arima(series: pd.Series, steps: int) -> np.ndarray:
    """Enhanced ARIMA forecasting with fallbacks"""
    if PMDARIMA_AVAILABLE:
        try:
            model = pm.auto_arima(
                series, 
                seasonal=False, 
                stepwise=True, 
                suppress_warnings=True, 
                error_action='ignore',
                max_p=3, max_q=3, max_d=2  # Limit complexity
            )
            forecast = model.predict(n_periods=steps)
            return forecast
        except Exception as e:
            st.warning(f"Auto-ARIMA failed, trying simple ARIMA: {str(e)}")
    
    return simple_arima_forecast(series, steps)

@st.cache_data(show_spinner=False)
def cached_forecast_prophet(df: pd.DataFrame, time_col: str, value_col: str, 
                           periods: int, holidays: Optional[pd.DataFrame] = None,
                           custom_seasonality: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Enhanced Prophet forecasting"""
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not available")
    
    try:
        df_prophet = df[[time_col, value_col]].rename(columns={time_col: 'ds', value_col: 'y'})
        df_prophet = df_prophet.dropna()
        
        if len(df_prophet) < 10:
            raise ValueError("Insufficient data for Prophet (need at least 10 points)")
        
        # Enhanced Prophet configuration
        model = Prophet(
            holidays=holidays,
            daily_seasonality='auto' if len(df_prophet) > 730 else False,
            weekly_seasonality='auto' if len(df_prophet) > 14 else False,
            yearly_seasonality='auto' if len(df_prophet) > 730 else False,
            changepoint_prior_scale=0.05,  # More flexible
            seasonality_prior_scale=10.0,
            interval_width=0.8
        )
        
        if custom_seasonality:
            for season in custom_seasonality:
                model.add_seasonality(
                    name=season.get('name', 'custom'),
                    period=season.get('period', 30.5),
                    fourier_order=season.get('fourier_order', 5)
                )
        
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
    except Exception as e:
        raise Exception(f"Prophet forecasting failed: {str(e)}")

# -------------------------------
# Enhanced Anomaly Detection
# -------------------------------
def detect_anomalies_comprehensive(series: pd.Series) -> Dict[str, pd.Series]:
    """Comprehensive anomaly detection with multiple methods"""
    anomalies = {}
    
    try:
        # Statistical method (Z-score)
        mean_val = series.mean()
        std_val = series.std()
        if std_val > 0:
            z_scores = np.abs((series - mean_val) / std_val)
            anomalies['Statistical'] = series[z_scores > 3]
        else:
            anomalies['Statistical'] = pd.Series(dtype=float)
    except Exception as e:
        st.warning(f"Statistical anomaly detection failed: {str(e)}")
        anomalies['Statistical'] = pd.Series(dtype=float)
    
    try:
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies['IQR'] = series[(series < lower_bound) | (series > upper_bound)]
    except Exception as e:
        st.warning(f"IQR anomaly detection failed: {str(e)}")
        anomalies['IQR'] = pd.Series(dtype=float)
    
    try:
        # Isolation Forest
        if len(series) >= 10:  # Need minimum samples
            clf = IsolationForest(contamination=0.1, random_state=42)
            data = series.values.reshape(-1, 1)
            clf.fit(data)
            pred = clf.predict(data)
            anomalies['Isolation Forest'] = series[pred == -1]
        else:
            anomalies['Isolation Forest'] = pd.Series(dtype=float)
    except Exception as e:
        st.warning(f"Isolation Forest anomaly detection failed: {str(e)}")
        anomalies['Isolation Forest'] = pd.Series(dtype=float)
    
    return anomalies

# -------------------------------
# Enhanced Plotting Functions
# -------------------------------
def create_forecast_plot(df: pd.DataFrame, forecast_data: Any, time_col: str, 
                        metric: str, model_name: str) -> go.Figure:
    """Create enhanced forecast visualization"""
    try:
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[metric],
            mode='lines',
            name='Historical Data',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        if isinstance(forecast_data, pd.DataFrame) and 'yhat' in forecast_data.columns:
            # Prophet forecast
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='%{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ))
            
            if 'yhat_upper' in forecast_data.columns and 'yhat_lower' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='rgba(255,127,14,0.3)', width=1),
                    showlegend=False,
                    hovertemplate='%{x}<br>Upper: %{y:.2f}<extra></extra>'
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_lower'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(color='rgba(255,127,14,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,127,14,0.1)',
                    showlegend=False,
                    hovertemplate='%{x}<br>Lower: %{y:.2f}<extra></extra>'
                ))
        else:
            # ARIMA or other forecast (numpy array)
            future_dates = pd.date_range(
                start=df[time_col].iloc[-1] + pd.Timedelta(days=1),
                periods=len(forecast_data)
            )
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=forecast_data,
                mode='lines',
                name='Forecast',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                hovertemplate='%{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>{model_name} Forecast for {metric}</b>",
                font=dict(size=18, color='#1e293b'),
                x=0.5
            ),
            xaxis_title=time_col,
            yaxis_title=metric,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating forecast plot: {str(e)}")
        return go.Figure()

def create_anomaly_plot(df: pd.DataFrame, time_col: str, metric: str, 
                       anomalies_dict: Dict[str, pd.Series]) -> go.Figure:
    """Create enhanced anomaly visualization"""
    try:
        fig = go.Figure()
        
        # Main data line
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[metric],
            mode='lines',
            name='Normal Data',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='%{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (method, anomalies) in enumerate(anomalies_dict.items()):
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies.values,
                    mode='markers',
                    name=f'{method} Anomalies ({len(anomalies)})',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=10,
                        symbol='x',
                        line=dict(width=2)
                    ),
                    hovertemplate=f'{method}<br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text=f"<b>Anomaly Detection for {metric}</b>",
                font=dict(size=18, color='#1e293b'),
                x=0.5
            ),
            xaxis_title=time_col,
            yaxis_title=metric,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating anomaly plot: {str(e)}")
        return go.Figure()

# -------------------------------
# Enhanced Model Performance
# -------------------------------
def compute_enhanced_metrics(actual: pd.Series, forecast: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive model metrics"""
    try:
        min_len = min(len(actual), len(forecast))
        if min_len == 0:
            return {'RMSE': float('inf'), 'MAPE': float('inf'), 'MAE': float('inf'), 'R²': -1}
        
        actual_subset = actual.iloc[-min_len:].values
        forecast_subset = forecast[-min_len:] if len(forecast) >= min_len else forecast
        
        rmse = mean_squared_error(actual_subset, forecast_subset, squared=False)
        mape = mean_absolute_percentage_error(actual_subset, forecast_subset) * 100
        mae = np.mean(np.abs(actual_subset - forecast_subset))
        
        # R-squared
        ss_res = np.sum((actual_subset - forecast_subset) ** 2)
        ss_tot = np.sum((actual_subset - np.mean(actual_subset)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else -1
        
        return {'RMSE': rmse, 'MAPE': mape, 'MAE': mae, 'R²': r2}
    except Exception as e:
        st.warning(f"Error computing metrics: {str(e)}")
        return {'RMSE': float('inf'), 'MAPE': float('inf'), 'MAE': float('inf'), 'R²': -1}

# -------------------------------
# Enhanced Insights Generation
# -------------------------------
def generate_enhanced_insights(df: pd.DataFrame, metric: str, 
                              anomalies_dict: Dict[str, pd.Series],
                              forecast_results: Dict[str, Any]) -> List[str]:
    """Generate comprehensive insights"""
    insights = []
    try:
        series = df[metric]
        
        # Trend analysis
        if len(series) >= 30:
            recent_trend = series.tail(30).mean()
            historical_avg = series.mean()
            if recent_trend > historical_avg * 1.1:
                insights.append(f"📈 {metric} shows an upward trend (recent avg: {recent_trend:.2f} vs historical: {historical_avg:.2f})")
            elif recent_trend < historical_avg * 0.9:
                insights.append(f"📉 {metric} shows a downward trend (recent avg: {recent_trend:.2f} vs historical: {historical_avg:.2f})")
            else:
                insights.append(f"➡️ {metric} remains stable around its historical average ({historical_avg:.2f})")
        
        # Volatility analysis
        volatility = series.std()
        cv = volatility / series.mean() if series.mean() != 0 else float('inf')
        if cv > 0.5:
            insights.append(f"⚠️ {metric} shows high volatility (CV: {cv:.2f})")
        elif cv < 0.1:
            insights.append(f"✅ {metric} shows low volatility (CV: {cv:.2f})")
        
        # Seasonality detection (basic)
        if len(series) >= 365:
            try:
                if STATSMODELS_AVAILABLE:
                    decomposition = seasonal_decompose(series, model='additive', period=30)
                    seasonal_strength = decomposition.seasonal.std() / series.std()
                    if seasonal_strength > 0.1:
                        insights.append(f"🔄 {metric} shows seasonal patterns (strength: {seasonal_strength:.2f})")
            except:
                pass
        
        # Anomaly insights
        total_anomalies = sum(len(anomalies) for anomalies in anomalies_dict.values())
        anomaly_rate = total_anomalies / len(series) * 100
        if anomaly_rate > 5:
            insights.append(f"🚨 High anomaly rate detected: {anomaly_rate:.1f}% of data points")
        elif anomaly_rate > 0:
            insights.append(f"ℹ️ {total_anomalies} anomalies detected ({anomaly_rate:.1f}% of data)")
        
        # Forecast quality insights
        if forecast_results:
            best_model = min(forecast_results.keys(), 
                           key=lambda x: forecast_results[x].get('RMSE', float('inf')))
            best_rmse = forecast_results[best_model]['RMSE']
            data_std = series.std()
            if best_rmse < data_std * 0.5:
                insights.append(f"✅ Excellent forecast quality with {best_model} (RMSE: {best_rmse:.2f})")
            elif best_rmse < data_std:
                insights.append(f"👍 Good forecast quality with {best_model} (RMSE: {best_rmse:.2f})")
            else:
                insights.append(f"⚠️ Moderate forecast quality with {best_model} (RMSE: {best_rmse:.2f})")
    
    except Exception as e:
        insights.append(f"❌ Error generating insights: {str(e)}")
    
    return insights

# -------------------------------
# Threading Function for Forecasting
# -------------------------------
def run_comprehensive_forecast(metric: str, df: pd.DataFrame, time_col: str, 
                              horizon: int, models_selected: List[str],
                              holidays_df: Optional[pd.DataFrame],
                              custom_seasonality: Optional[List[Dict]]) -> Tuple[str, Dict[str, Any]]:
    """Run comprehensive forecasting for a metric"""
    results = {}
    
    # Auto-ARIMA or Simple ARIMA
    if "ARIMA" in models_selected:
        try:
            arima_forecast = cached_forecast_arima(df[metric], horizon)
            metrics = compute_enhanced_metrics(df[metric], arima_forecast)
            results['ARIMA'] = {
                'forecast': arima_forecast,
                **metrics
            }
        except Exception as e:
            st.warning(f"ARIMA failed for {metric}: {str(e)}")
    
    # Prophet
    if "Prophet" in models_selected and PROPHET_AVAILABLE:
        try:
            prophet_forecast = cached_forecast_prophet(
                df, time_col, metric, horizon,
                holidays=holidays_df,
                custom_seasonality=custom_seasonality
            )
            
            # Calculate metrics on overlapping period
            if len(df[metric]) >= horizon:
                y_true = df[metric].values[-horizon:]
                y_pred = prophet_forecast['yhat'].values[-horizon:]
                metrics = compute_enhanced_metrics(pd.Series(y_true), y_pred)
            else:
                metrics = {'RMSE': float('inf'), 'MAPE': float('inf'), 'MAE': float('inf'), 'R²': -1}
            
            results['Prophet'] = {
                'forecast': prophet_forecast,
                **metrics
            }
        except Exception as e:
            st.warning(f"Prophet failed for {metric}: {str(e)}")
    
    # Linear Regression (simple baseline)
    if "Linear Trend" in models_selected:
        try:
            X = np.arange(len(df[metric])).reshape(-1, 1)
            y = df[metric].values
            model = LinearRegression().fit(X, y)
            future_X = np.arange(len(df[metric]), len(df[metric]) + horizon).reshape(-1, 1)
            linear_forecast = model.predict(future_X)
            metrics = compute_enhanced_metrics(df[metric], linear_forecast)
            results['Linear Trend'] = {
                'forecast': linear_forecast,
                **metrics
            }
        except Exception as e:
            st.warning(f"Linear Trend failed for {metric}: {str(e)}")
    
    return metric, results

# -------------------------------
# Main Application
# -------------------------------
def main():
    # Load CSS
    load_css()
    
    # Main Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 AI Forecasting Pro</h1>
        <p>Advanced Time Series Forecasting & Anomaly Detection Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check available models
    available_models = []
    if PMDARIMA_AVAILABLE or STATSMODELS_AVAILABLE:
        available_models.append("ARIMA")
    if PROPHET_AVAILABLE:
        available_models.append("Prophet")
    available_models.append("Linear Trend")  # Always available
    
    if not available_models:
        st.error("⚠️ No forecasting models available. Please check your installation.")
        st.stop()
    
    # Display model availability
    col1, col2, col3 = st.columns(3)
    with col1:
        prophet_status = "✅ Available" if PROPHET_AVAILABLE else "❌ Not Available"
        st.markdown(f"**Prophet:** <span class='{'status-success' if PROPHET_AVAILABLE else 'status-error'}'>{prophet_status}</span>", unsafe_allow_html=True)
    
    with col2:
        arima_status = "✅ Available" if (PMDARIMA_AVAILABLE or STATSMODELS_AVAILABLE) else "❌ Not Available"
        st.markdown(f"**ARIMA:** <span class='{'status-success' if (PMDARIMA_AVAILABLE or STATSMODELS_AVAILABLE) else 'status-error'}'>{arima_status}</span>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Linear Trend:** <span class='status-success'>✅ Available</span>", unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.markdown("## 🔧 Configuration")
    
    # File Upload
    st.sidebar.markdown("### 📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Choose your time series data file",
        type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel file with time series data"
    )
    
    if not uploaded_file:
        # Demo data option
        st.markdown("---")
        if st.button("🚀 Try with Demo Data", help="Load sample data to explore features"):
            # Create sample data
            dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
            np.random.seed(42)
            trend = np.linspace(100, 150, len(dates))
            seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
            noise = np.random.normal(0, 5, len(dates))
            values = trend + seasonal + noise
            
            demo_df = pd.DataFrame({
                'date': dates,
                'sales': values,
                'inventory': values * 0.8 + np.random.normal(0, 3, len(dates)),
                'temperature': 20 + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + np.random.normal(0, 2, len(dates))
            })
            
            st.session_state['demo_data'] = demo_df
            st.success("✅ Demo data loaded! Check the sidebar to configure your analysis.")
        
        if 'demo_data' not in st.session_state:
            st.info("👆 Please upload your time series data file or try the demo data to get started.")
            return
        else:
            df = st.session_state['demo_data']
            st.info("📊 Using demo data. Upload your own file to analyze real data.")
    else:
        df = load_data(uploaded_file)
        if df is None:
            st.stop()
    
    # Data preprocessing
    time_col_auto = detect_time_column(df)
    if time_col_auto is None:
        st.error("❌ No datetime column detected. Please ensure your data has a datetime column.")
        st.stop()
    
    # Sidebar configuration continues
    st.sidebar.markdown("### 📅 Time Configuration")
    time_col = st.sidebar.selectbox(
        "Select Time Column",
        df.columns,
        index=df.columns.get_loc(time_col_auto),
        help="Choose the column containing datetime values"
    )
    
    df = preprocess_data(df, time_col)
    
    # Metric selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("❌ No numeric columns found for forecasting.")
        st.stop()
    
    st.sidebar.markdown("### 📊 Metrics Selection")
    metrics_selected = st.sidebar.multiselect(
        "Select Metrics to Forecast",
        numeric_cols,
        default=numeric_cols[:min(3, len(numeric_cols))],
        help="Choose the metrics you want to forecast"
    )
    
    if not metrics_selected:
        st.warning("⚠️ Please select at least one metric to forecast.")
        st.stop()
    
    # Forecast configuration
    st.sidebar.markdown("### 🔮 Forecast Configuration")
    
    # Global or individual horizons
    use_global_horizon = st.sidebar.checkbox("Use same horizon for all metrics", value=True)
    horizon_dict = {}
    
    if use_global_horizon:
        global_horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 365, 30)
        horizon_dict = {metric: global_horizon for metric in metrics_selected}
    else:
        for metric in metrics_selected:
            horizon_dict[metric] = st.sidebar.slider(
                f"{metric} horizon (days)", 1, 365, 30, key=f"horizon_{metric}"
            )
    
    # Model selection
    st.sidebar.markdown("### 🤖 Model Selection")
    models_selected = st.sidebar.multiselect(
        "Select Forecasting Models",
        available_models,
        default=available_models[:2] if len(available_models) >= 2 else available_models,
        help="Choose which models to use for forecasting"
    )
    
    if not models_selected:
        st.warning("⚠️ Please select at least one forecasting model.")
        st.stop()
    
    # Advanced configurations
    with st.sidebar.expander("🔧 Advanced Settings"):
        # Holidays for Prophet
        st.markdown("**Holidays Configuration (Prophet only)**")
        holiday_file = st.file_uploader(
            "Upload Holidays CSV",
            type=["csv"],
            help="CSV file with 'ds' column for holiday dates"
        )
        holidays_df = None
        if holiday_file and PROPHET_AVAILABLE:
            try:
                holidays_df = pd.read_csv(holiday_file)
                if 'ds' in holidays_df.columns:
                    holidays_df['ds'] = pd.to_datetime(holidays_df['ds'])
                    st.success(f"✅ Loaded {len(holidays_df)} holidays")
                else:
                    st.warning("⚠️ Holiday file must have a 'ds' column")
                    holidays_df = None
            except Exception as e:
                st.error(f"❌ Error reading holidays: {str(e)}")
        
        # Custom seasonality for Prophet
        st.markdown("**Custom Seasonality (Prophet only)**")
        custom_seasonality_input = st.text_area(
            "Enter as JSON",
            placeholder='[{"name":"monthly","period":30.5,"fourier_order":5}]',
            help="JSON array of custom seasonality configurations"
        )
        custom_seasonality = None
        if custom_seasonality_input.strip() and PROPHET_AVAILABLE:
            try:
                custom_seasonality = json.loads(custom_seasonality_input)
                st.success(f"✅ Loaded {len(custom_seasonality)} custom seasonalities")
            except Exception as e:
                st.warning(f"⚠️ Invalid JSON: {str(e)}")
    
    # Main content tabs
    tabs = st.tabs([
        "📊 Data Overview",
        "🔮 Forecasting",
        "🚨 Anomaly Detection", 
        "💡 Insights",
        "📈 Model Comparison",
        "💾 Export Results"
    ])
    
    # --- Data Overview Tab ---
    with tabs[0]:
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Records</div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Date Range</div>
            </div>
            """.format(f"{df[time_col].min().date()} to {df[time_col].max().date()}"), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Numeric Columns</div>
            </div>
            """.format(len(numeric_cols)), unsafe_allow_html=True)
        
        with col4:
            missing_pct = (df[metrics_selected].isnull().sum().sum() / (len(df) * len(metrics_selected)) * 100)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Missing Data</div>
            </div>
            """.format(missing_pct), unsafe_allow_html=True)
        
        st.markdown("### 📈 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("### 📋 Statistical Summary")
        st.dataframe(df[metrics_selected].describe(), use_container_width=True)
        
        # Data visualization
        if len(metrics_selected) <= 4:
            st.markdown("### 📊 Time Series Visualization")
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(metrics_selected):
                fig.add_trace(go.Scatter(
                    x=df[time_col],
                    y=df[metric],
                    mode='lines',
                    name=metric,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'{metric}<br>%{{x}}<br>Value: %{{y:.2f}}<extra></extra>'
                ))
            
            fig.update_layout(
                title="Time Series Overview",
                xaxis_title=time_col,
                yaxis_title="Values",
                template='plotly_white',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # --- Forecasting Tab ---
    with tabs[1]:
        st.markdown("### 🔮 Forecasting Results")
        
        forecast_results = {}
        
        if st.button("🚀 Run Forecasting", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with ThreadPoolExecutor(max_workers=min(4, len(metrics_selected))) as executor:
                futures = []
                for i, metric in enumerate(metrics_selected):
                    future = executor.submit(
                        run_comprehensive_forecast,
                        metric, df, time_col, horizon_dict[metric],
                        models_selected, holidays_df, custom_seasonality
                    )
                    futures.append(future)
                
                for i, future in enumerate(futures):
                    status_text.text(f"Processing {metrics_selected[i]}...")
                    metric, results = future.result()
                    forecast_results[metric] = results
                    progress_bar.progress((i + 1) / len(metrics_selected))
            
            status_text.text("✅ Forecasting completed!")
            st.session_state['forecast_results'] = forecast_results
        
        # Display results if available
        if 'forecast_results' in st.session_state:
            forecast_results = st.session_state['forecast_results']
            
            for metric in metrics_selected:
                if metric not in forecast_results or not forecast_results[metric]:
                    continue
                
                st.markdown(f"#### 📊 {metric} Forecast Results")
                
                results = forecast_results[metric]
                
                # Model comparison table
                if results:
                    comparison_data = []
                    for model_name, model_results in results.items():
                        comparison_data.append({
                            'Model': model_name,
                            'RMSE': f"{model_results['RMSE']:.2f}",
                            'MAPE': f"{model_results['MAPE']:.1f}%",
                            'MAE': f"{model_results['MAE']:.2f}",
                            'R²': f"{model_results['R²']:.3f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Find best model
                    best_model_idx = min(range(len(comparison_data)), 
                                       key=lambda i: results[comparison_data[i]['Model']]['RMSE'])
                    best_model_name = comparison_data[best_model_idx]['Model']
                    
                    st.markdown(f"**🏆 Best Model: {best_model_name}**")
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Plot forecast
                    best_forecast = results[best_model_name]['forecast']
                    fig = create_forecast_plot(df, best_forecast, time_col, metric, best_model_name)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Key metrics display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_value = df[metric].iloc[-1]
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{current_value:.2f}</div>
                            <div class="metric-label">Current Value</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if isinstance(best_forecast, pd.DataFrame) and 'yhat' in best_forecast.columns:
                            forecast_value = best_forecast['yhat'].iloc[-1]
                        else:
                            forecast_value = best_forecast[-1] if len(best_forecast) > 0 else 0
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{forecast_value:.2f}</div>
                            <div class="metric-label">Forecast Value</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        change_pct = ((forecast_value - current_value) / current_value * 100) if current_value != 0 else 0
                        change_color = "#10b981" if change_pct >= 0 else "#ef4444"
                        change_icon = "📈" if change_pct >= 0 else "📉"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value" style="color: {change_color};">{change_icon} {change_pct:+.1f}%</div>
                            <div class="metric-label">Predicted Change</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    # --- Anomaly Detection Tab ---
    with tabs[2]:
        st.markdown("### 🚨 Anomaly Detection")
        
        if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
            anomaly_results = {}
            progress_bar = st.progress(0)
            
            for i, metric in enumerate(metrics_selected):
                anomalies_dict = detect_anomalies_comprehensive(df[metric])
                anomaly_results[metric] = anomalies_dict
                progress_bar.progress((i + 1) / len(metrics_selected))
            
            st.session_state['anomaly_results'] = anomaly_results
        
        if 'anomaly_results' in st.session_state:
            anomaly_results = st.session_state['anomaly_results']
            
            for metric in metrics_selected:
                if metric not in anomaly_results:
                    continue
                
                st.markdown(f"#### 🔍 {metric} Anomaly Analysis")
                
                anomalies_dict = anomaly_results[metric]
                
                # Anomaly summary
                total_anomalies = sum(len(anomalies) for anomalies in anomalies_dict.values())
                anomaly_rate = total_anomalies / len(df) * 100 if len(df) > 0 else 0
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{total_anomalies}</div>
                        <div class="metric-label">Total Anomalies</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{anomaly_rate:.1f}%</div>
                        <div class="metric-label">Anomaly Rate</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    severity = "High" if anomaly_rate > 5 else "Medium" if anomaly_rate > 2 else "Low"
                    severity_color = "#ef4444" if severity == "High" else "#f59e0b" if severity == "Medium" else "#10b981"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: {severity_color};">{severity}</div>
                        <div class="metric-label">Severity Level</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Anomaly plot
                fig = create_anomaly_plot(df, time_col, metric, anomalies_dict)
                st.plotly_chart(fig, use_container_width=True)
                
                # Anomaly details
                if total_anomalies > 0:
                    with st.expander(f"📋 Detailed Anomaly List for {metric}"):
                        for method, anomalies in anomalies_dict.items():
                            if len(anomalies) > 0:
                                st.markdown(f"**{method} Method ({len(anomalies)} anomalies):**")
                                anomaly_df = pd.DataFrame({
                                    'Date': anomalies.index,
                                    'Value': anomalies.values,
                                    'Method': method
                                })
                                st.dataframe(anomaly_df, use_container_width=True)
                
                st.markdown("---")
    
    # --- Insights Tab ---
    with tabs[3]:
        st.markdown("### 💡 AI-Generated Insights")
        
        if 'forecast_results' in st.session_state and 'anomaly_results' in st.session_state:
            for metric in metrics_selected:
                forecast_data = st.session_state['forecast_results'].get(metric, {})
                anomaly_data = st.session_state['anomaly_results'].get(metric, {})
                
                insights = generate_enhanced_insights(df, metric, anomaly_data, forecast_data)
                
                if insights:
                    st.markdown(f"#### 🎯 Insights for {metric}")
                    
                    for insight in insights:
                        st.markdown(f"""
                        <div class="metric-card">
                            <p style="margin: 0; font-size: 1rem; line-height: 1.5;">{insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
        else:
            st.info("📊 Run forecasting and anomaly detection first to generate insights.")
    
    # --- Model Comparison Tab ---
    with tabs[4]:
        st.markdown("### 📈 Model Performance Comparison")
        
        if 'forecast_results' in st.session_state:
            forecast_results = st.session_state['forecast_results']
            
            # Prepare comparison data
            comparison_data = []
            for metric in metrics_selected:
                if metric in forecast_results:
                    for model_name, model_results in forecast_results[metric].items():
                        comparison_data.append({
                            'Metric': metric,
                            'Model': model_name,
                            'RMSE': model_results['RMSE'],
                            'MAPE': model_results['MAPE'],
                            'MAE': model_results['MAE'],
                            'R²': model_results['R²']
                        })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Overall best models
                st.markdown("#### 🏆 Best Models by Metric")
                best_models = []
                for metric in metrics_selected:
                    metric_data = comparison_df[comparison_df['Metric'] == metric]
                    if not metric_data.empty:
                        best_idx = metric_data['RMSE'].idxmin()
                        best_models.append({
                            'Metric': metric,
                            'Best Model': metric_data.loc[best_idx, 'Model'],
                            'RMSE': metric_data.loc[best_idx, 'RMSE'],
                            'MAPE': f"{metric_data.loc[best_idx, 'MAPE']:.1f}%",
                            'R²': f"{metric_data.loc[best_idx, 'R²']:.3f}"
                        })
                
                st.dataframe(pd.DataFrame(best_models), use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_rmse = px.bar(
                        comparison_df, 
                        x='Metric', 
                        y='RMSE', 
                        color='Model',
                        title="RMSE Comparison by Model",
                        template='plotly_white'
                    )
                    fig_rmse.update_layout(height=400)
                    st.plotly_chart(fig_rmse, use_container_width=True)
                
                with col2:
                    fig_mape = px.bar(
                        comparison_df, 
                        x='Metric', 
                        y='MAPE', 
                        color='Model',
                        title="MAPE Comparison by Model",
                        template='plotly_white'
                    )
                    fig_mape.update_layout(height=400)
                    st.plotly_chart(fig_mape, use_container_width=True)
                
                # Detailed comparison table
                st.markdown("#### 📊 Detailed Performance Metrics")
                st.dataframe(comparison_df, use_container_width=True)
            
        else:
            st.info("📊 Run forecasting first to compare model performance.")
    
    # --- Export Results Tab ---
    with tabs[5]:
        st.markdown("### 💾 Export Results")
        
        if 'forecast_results' in st.session_state:
            forecast_results = st.session_state['forecast_results']
            
            st.markdown("#### 📥 Download Forecast Data")
            
            for metric in metrics_selected:
                if metric in forecast_results and forecast_results[metric]:
                    st.markdown(f"**{metric} Forecasts:**")
                    
                    # Find best model
                    results = forecast_results[metric]
                    best_model = min(results.keys(), key=lambda x: results[x]['RMSE'])
                    forecast_data = results[best_model]['forecast']
                    
                    col1, col2 = st.columns(2)
                    
                    # Prepare download data
                    if isinstance(forecast_data, pd.DataFrame):
                        download_df = forecast_data.copy()
                    else:
                        # Convert numpy array to DataFrame
                        future_dates = pd.date_range(
                            start=df[time_col].iloc[-1] + pd.Timedelta(days=1),
                            periods=len(forecast_data)
                        )
                        download_df = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': forecast_data
                        })
                    
                    with col1:
                        csv_data = download_df.to_csv(index=False).encode()
                        st.download_button(
                            label=f"📄 Download {metric} CSV",
                            data=csv_data,
                            file_name=f"{metric}_forecast_{best_model.lower()}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Excel export
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            download_df.to_excel(writer, sheet_name=f'{metric}_Forecast', index=False)
                        
                        st.download_button(
                            label=f"📊 Download {metric} Excel",
                            data=excel_buffer.getvalue(),
                            file_name=f"{metric}_forecast_{best_model.lower()}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    st.markdown("---")
            
            # Export all results as JSON
            st.markdown("#### 📋 Export Complete Analysis")
            if st.button("📦 Prepare Complete Export", use_container_width=True):
                export_data = {
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'metrics_analyzed': metrics_selected,
                        'models_used': models_selected,
                        'data_range': {
                            'start': df[time_col].min().isoformat(),
                            'end': df[time_col].max().isoformat(),
                            'records': len(df)
                        }
                    },
                    'forecast_results': {}
                }
                
                # Convert forecast results to serializable format
                for metric, results in forecast_results.items():
                    export_data['forecast_results'][metric] = {}
                    for model_name, model_results in results.items():
                        export_data['forecast_results'][metric][model_name] = {
                            'metrics': {
                                'RMSE': model_results['RMSE'],
                                'MAPE': model_results['MAPE'],
                                'MAE': model_results['MAE'],
                                'R²': model_results['R²']
                            }
                        }
                
                json_data = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📄 Download Complete Analysis (JSON)",
                    data=json_data.encode(),
                    file_name=f"forecasting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.info("📊 Run forecasting analysis first to export results.")

# -------------------------------
# Footer
# -------------------------------
def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p>📈 <strong>AI Forecasting Pro</strong> - Advanced Time Series Analysis Dashboard</p>
        <p style="font-size: 0.9rem;">
            Powered by Prophet, ARIMA, and Machine Learning • Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Requirements.txt Generator
# -------------------------------
def show_requirements_info():
    """Show requirements information for deployment"""
    if st.sidebar.button("📋 Show Requirements.txt"):
        requirements_content = """streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
prophet>=1.1.4
pmdarima>=2.0.0
statsmodels>=0.14.0
"""
        st.sidebar.code(requirements_content, language="text")
        st.sidebar.info("💡 Copy this content to your requirements.txt file for deployment")

if __name__ == "__main__":
    main()
    show_requirements_info()
    show_footer()
