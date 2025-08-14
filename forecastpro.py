import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
import warnings
import io
import json
from concurrent.futures import ThreadPoolExecutor

# Import libraries with fallbacks
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not available. Charts will be disabled.")

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("Scikit-learn not available. Some features will be limited.")

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

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Time Series Forecasting Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    .status-success {
        color: #10b981;
        font-weight: bold;
    }
    .status-error {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
@st.cache_data
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def detect_datetime_column(df):
    """Automatically detect datetime columns"""
    datetime_cols = []
    for col in df.columns:
        try:
            # Try to convert a sample of the column
            sample = df[col].dropna().head(100)
            pd.to_datetime(sample, errors='raise')
            datetime_cols.append(col)
        except:
            continue
    return datetime_cols

def preprocess_data(df, time_col):
    """Clean and preprocess the data"""
    try:
        # Convert to datetime
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Remove rows where datetime conversion failed
        df = df.dropna(subset=[time_col])
        
        # Sort by time
        df = df.sort_values(time_col).reset_index(drop=True)
        
        return df
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return df

# Simple Moving Average Forecast (fallback when other libraries fail)
def simple_moving_average_forecast(series, window=7, periods=7):
    """Simple moving average forecast"""
    try:
        if len(series) < window:
            window = len(series)
        
        moving_avg = series.rolling(window=window).mean().iloc[-1]
        forecast = [moving_avg] * periods
        return np.array(forecast)
    except:
        return np.array([series.mean()] * periods)

# Linear trend forecast (fallback method)
def linear_trend_forecast(series, periods=7):
    """Simple linear trend extrapolation"""
    try:
        x = np.arange(len(series))
        y = series.values
        
        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Generate forecast
        future_x = np.arange(len(series), len(series) + periods)
        forecast = slope * future_x + intercept
        
        return forecast
    except:
        return np.array([series.mean()] * periods)

# Anomaly detection with simple statistical method
def detect_anomalies_simple(series, threshold=3):
    """Detect anomalies using z-score method"""
    try:
        z_scores = np.abs((series - series.mean()) / series.std())
        anomalies = series[z_scores > threshold]
        return anomalies
    except:
        return pd.Series(dtype=float)

def detect_anomalies_iqr(series):
    """Detect anomalies using IQR method"""
    try:
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = series[(series < lower_bound) | (series > upper_bound)]
        return anomalies
    except:
        return pd.Series(dtype=float)

# Advanced forecasting methods (when libraries are available)
@st.cache_data(show_spinner=False)
def prophet_forecast(df, time_col, value_col, periods):
    """Prophet forecasting method"""
    if not PROPHET_AVAILABLE:
        return None
    
    try:
        # Prepare data for Prophet
        prophet_df = df[[time_col, value_col]].rename(columns={time_col: 'ds', value_col: 'y'})
        prophet_df = prophet_df.dropna()
        
        # Initialize and fit model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            interval_width=0.95
        )
        model.fit(prophet_df)
        
        # Make future predictions
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        st.warning(f"Prophet forecasting failed: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def arima_forecast(series, periods):
    """ARIMA forecasting method"""
    if not PMDARIMA_AVAILABLE:
        return None
    
    try:
        model = pm.auto_arima(
            series,
            start_p=1, start_q=1,
            max_p=3, max_q=3,
            seasonal=False,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        forecast = model.predict(n_periods=periods)
        return forecast
    except Exception as e:
        st.warning(f"ARIMA forecasting failed: {str(e)}")
        return None

# Plotting functions
def create_basic_plot(df, time_col, value_col, forecast_data=None, anomalies=None, title="Time Series"):
    """Create basic plot using Plotly or fallback to line chart"""
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        # Add actual data
        fig.add_trace(go.Scatter(
            x=df[time_col],
            y=df[value_col],
            mode='lines',
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add forecast if available
        if forecast_data is not None:
            if isinstance(forecast_data, pd.DataFrame) and 'ds' in forecast_data.columns:
                # Prophet forecast
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_upper'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(color='lightcoral', dash='dot'),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'],
                    y=forecast_data['yhat_lower'],
                    mode='lines',
                    name='Lower Bound',
                    line=dict(color='lightcoral', dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(255,182,193,0.2)',
                    showlegend=False
                ))
            else:
                # Simple forecast (array)
                last_date = df[time_col].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(forecast_data))
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=forecast_data,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
        
        # Add anomalies if available
        if anomalies is not None and len(anomalies) > 0:
            anomaly_dates = df.loc[anomalies.index, time_col]
            fig.add_trace(go.Scatter(
                x=anomaly_dates,
                y=anomalies.values,
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    else:
        # Fallback to Streamlit line chart
        chart_data = df.set_index(time_col)[value_col]
        return chart_data

# Main Application
def main():
    st.markdown('<h1 class="main-header">📈 Time Series Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Show library status
    with st.expander("📋 Library Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Core Libraries:**")
            st.markdown(f"• Pandas: ✅")
            st.markdown(f"• NumPy: ✅")
            st.markdown(f"• Plotly: {'✅' if PLOTLY_AVAILABLE else '❌'}")
            
        with col2:
            st.markdown("**ML Libraries:**")
            st.markdown(f"• Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
            st.markdown(f"• Prophet: {'✅' if PROPHET_AVAILABLE else '❌'}")
            st.markdown(f"• pmdarima: {'✅' if PMDARIMA_AVAILABLE else '❌'}")
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your time series data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file with time series data"
    )
    
    if uploaded_file is None:
        st.info("👆 Please upload a CSV or Excel file to get started!")
        
        # Show example data format
        st.subheader("📊 Expected Data Format")
        example_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.random.normal(1000, 100, 100) + np.sin(np.arange(100) * 0.1) * 50,
            'revenue': np.random.normal(5000, 500, 100) + np.cos(np.arange(100) * 0.1) * 200
        })
        st.dataframe(example_data.head(10))
        
        # Download example data
        csv = example_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Example Data",
            data=csv,
            file_name="example_timeseries_data.csv",
            mime="text/csv"
        )
        return
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data(uploaded_file)
    
    if df is None:
        return
    
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Detect datetime columns
    datetime_cols = detect_datetime_column(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not datetime_cols:
        st.error("❌ No datetime columns detected in your data!")
        st.stop()
    
    if not numeric_cols:
        st.error("❌ No numeric columns detected in your data!")
        st.stop()
    
    # Column selection
    time_col = st.sidebar.selectbox("📅 Select Time Column", datetime_cols)
    metrics = st.sidebar.multiselect("📊 Select Metrics to Analyze", numeric_cols, default=numeric_cols[:3])
    
    if not metrics:
        st.warning("⚠️ Please select at least one metric to analyze!")
        return
    
    # Preprocess data
    df = preprocess_data(df, time_col)
    
    # Forecast settings
    st.sidebar.subheader("🔮 Forecast Settings")
    forecast_days = st.sidebar.slider("Forecast Period (days)", 1, 90, 14)
    
    # Available methods
    available_methods = ["Moving Average", "Linear Trend"]
    if PROPHET_AVAILABLE:
        available_methods.append("Prophet")
    if PMDARIMA_AVAILABLE:
        available_methods.append("ARIMA")
    
    forecast_method = st.sidebar.selectbox("Forecasting Method", available_methods)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔮 Forecasting", "🚨 Anomaly Detection", "📈 Advanced Analysis"])
    
    with tab1:
        st.subheader("📊 Data Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Data Preview:**")
            st.dataframe(df.head(10))
            
        with col2:
            st.write("**Data Summary:**")
            st.write(f"• Total Records: {len(df):,}")
            st.write(f"• Date Range: {df[time_col].min().date()} to {df[time_col].max().date()}")
            st.write(f"• Metrics: {len(metrics)}")
        
        # Basic statistics
        st.write("**Statistical Summary:**")
        st.dataframe(df[metrics].describe())
        
        # Plot selected metrics
        for metric in metrics:
            if PLOTLY_AVAILABLE:
                fig = create_basic_plot(df, time_col, metric, title=f"Time Series: {metric}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                chart_data = df.set_index(time_col)[metric]
                st.line_chart(chart_data)
    
    with tab2:
        st.subheader("🔮 Forecasting Results")
        
        for metric in metrics:
            st.write(f"**Forecasting: {metric}**")
            
            series = df[metric].dropna()
            
            # Generate forecast based on selected method
            forecast_result = None
            
            if forecast_method == "Moving Average":
                forecast_result = simple_moving_average_forecast(series, periods=forecast_days)
            elif forecast_method == "Linear Trend":
                forecast_result = linear_trend_forecast(series, periods=forecast_days)
            elif forecast_method == "Prophet" and PROPHET_AVAILABLE:
                forecast_result = prophet_forecast(df, time_col, metric, forecast_days)
            elif forecast_method == "ARIMA" and PMDARIMA_AVAILABLE:
                forecast_result = arima_forecast(series, forecast_days)
            
            if forecast_result is not None:
                # Plot forecast
                if PLOTLY_AVAILABLE:
                    fig = create_basic_plot(df, time_col, metric, forecast_result, title=f"Forecast: {metric} ({forecast_method})")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    chart_data = df.set_index(time_col)[metric]
                    st.line_chart(chart_data)
                    st.write("Forecast values:", forecast_result if isinstance(forecast_result, (list, np.ndarray)) else "Complex forecast data")
                
                # Show forecast summary
                current_value = series.iloc[-1]
                if isinstance(forecast_result, pd.DataFrame) and 'yhat' in forecast_result.columns:
                    predicted_value = forecast_result['yhat'].iloc[-1]
                elif isinstance(forecast_result, (list, np.ndarray)):
                    predicted_value = forecast_result[-1]
                else:
                    predicted_value = current_value
                
                change_pct = ((predicted_value - current_value) / current_value) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Value", f"{current_value:.2f}")
                with col2:
                    st.metric("Predicted Value", f"{predicted_value:.2f}", f"{change_pct:.1f}%")
                with col3:
                    st.metric("Forecast Period", f"{forecast_days} days")
            
            st.markdown("---")
    
    with tab3:
        st.subheader("🚨 Anomaly Detection")
        
        for metric in metrics:
            st.write(f"**Anomalies in: {metric}**")
            
            series = df[metric].dropna()
            
            # Detect anomalies using multiple methods
            anomalies_zscore = detect_anomalies_simple(series, threshold=3)
            anomalies_iqr = detect_anomalies_iqr(series)
            
            # Combine anomalies
            all_anomalies = pd.concat([anomalies_zscore, anomalies_iqr]).drop_duplicates()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Z-Score Anomalies", len(anomalies_zscore))
            with col2:
                st.metric("IQR Anomalies", len(anomalies_iqr))
            
            if len(all_anomalies) > 0:
                if PLOTLY_AVAILABLE:
                    fig = create_basic_plot(df, time_col, metric, anomalies=all_anomalies, title=f"Anomalies: {metric}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    chart_data = df.set_index(time_col)[metric]
                    st.line_chart(chart_data)
                
                # Show anomaly details
                anomaly_details = df.loc[all_anomalies.index, [time_col, metric]]
                st.write("**Anomaly Details:**")
                st.dataframe(anomaly_details)
            else:
                st.info("No anomalies detected in this metric.")
            
            st.markdown("---")
    
    with tab4:
        st.subheader("📈 Advanced Analysis")
        
        # Correlation matrix
        if len(metrics) > 1:
            st.write("**Correlation Matrix:**")
            corr_matrix = df[metrics].corr()
            
            if PLOTLY_AVAILABLE:
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Metric Correlations"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(corr_matrix)
        
        # Trend analysis
        st.write("**Trend Analysis:**")
        for metric in metrics:
            series = df[metric].dropna()
            
            # Calculate basic trend metrics
            first_half = series[:len(series)//2].mean()
            second_half = series[len(series)//2:].mean()
            trend_direction = "↗️ Increasing" if second_half > first_half else "↘️ Decreasing"
            trend_magnitude = abs((second_half - first_half) / first_half * 100)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{metric} Trend", trend_direction)
            with col2:
                st.metric("Trend Magnitude", f"{trend_magnitude:.1f}%")
            with col3:
                st.metric("Volatility (CV)", f"{(series.std()/series.mean()*100):.1f}%")
        
        # Data export
        st.subheader("📥 Export Data")
        
        # Prepare export data
        export_data = df.copy()
        
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="📊 Download Processed Data (CSV)",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
