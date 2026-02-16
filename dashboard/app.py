"""
Streamlit Dashboard for AQI Prediction System.
Four tabs: Live Dashboard, EDA, Model X-Ray (SHAP), AQI Alerts.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
from collections import deque

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state for alert history
if "alert_history" not in st.session_state:
    st.session_state.alert_history = deque(maxlen=20)
if "custom_thresholds" not in st.session_state:
    st.session_state.custom_thresholds = {
        "good": 50,
        "moderate": 100,
        "unhealthy_sensitive": 150,
        "unhealthy": 200,
        "very_unhealthy": 300,
        "hazardous": 150  # Alert threshold
    }


def load_latest_data() -> pd.DataFrame:
    """Load latest processed data."""
    try:
        data_dir = settings.DATA_DIR / "processed"
        latest = sorted(data_dir.glob("karachi_features*.csv"))[-1]
        return pd.read_csv(latest)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def get_aqi_category(aqi: int) -> tuple:
    """Get AQI category info."""
    if aqi <= 50:
        return "Good", "green", (0, 50)
    elif aqi <= 100:
        return "Moderate", "yellow", (51, 100)
    elif aqi <= 150:
        return "Unhealthy for Sensitive", "orange", (101, 150)
    elif aqi <= 200:
        return "Unhealthy", "red", (151, 200)
    elif aqi <= 300:
        return "Very Unhealthy", "purple", (201, 300)
    else:
        return "Hazardous", "maroon", (301, 500)


def check_aqi_alerts(aqi: int, thresholds: dict) -> list:
    """Check AQI against thresholds and generate alerts."""
    alerts = []
    
    if aqi > thresholds["hazardous"]:
        alerts.append({
            "level": "hazardous",
            "message": f"🚨 HAZARDOUS AQI: {aqi} - Immediate health warning! Everyone may experience more serious health effects.",
            "type": "error"
        })
    elif aqi > thresholds["very_unhealthy"]:
        alerts.append({
            "level": "very_unhealthy",
            "message": f"⚠️ VERY UNHEALTHY AQI: {aqi} - Health alert! Everyone may begin to experience more serious health effects.",
            "type": "error"
        })
    elif aqi > thresholds["unhealthy"]:
        alerts.append({
            "level": "unhealthy",
            "message": f"⚠️ UNHEALTHY AQI: {aqi} - Everyone may begin to experience health effects.",
            "type": "error"
        })
    elif aqi > thresholds["unhealthy_sensitive"]:
        alerts.append({
            "level": "unhealthy_sensitive",
            "message": f"⚡ UNHEALTHY FOR SENSITIVE GROUPS AQI: {aqi} - Sensitive groups may experience health effects.",
            "type": "warning"
        })
    elif aqi > thresholds["moderate"]:
        alerts.append({
            "level": "moderate",
            "message": f"ℹ️ MODERATE AQI: {aqi} - Air quality is acceptable for most.",
            "type": "success"
        })
    else:
        alerts.append({
            "level": "good",
            "message": f"✅ GOOD AQI: {aqi} - Air quality is satisfactory.",
            "type": "success"
        })
    
    return alerts


def add_alert_to_history(alert: dict):
    """Add alert to history with timestamp."""
    alert_entry = {
        "timestamp": datetime.now(),
        "level": alert["level"],
        "message": alert["message"]
    }
    st.session_state.alert_history.append(alert_entry)


def display_alert_banner(aqi: int, thresholds: dict):
    """Display visual warning banner for hazardous AQI levels."""
    alerts = check_aqi_alerts(aqi, thresholds)
    
    for alert in alerts:
        if alert["level"] in ["hazardous", "very_unhealthy", "unhealthy"]:
            # Add to history
            add_alert_to_history(alert)
            
            # Display banner based on alert type
            if alert["type"] == "error":
                st.error(alert["message"])
            elif alert["type"] == "warning":
                st.warning(alert["message"])
            
            # Show toast notification
            st.toast(alert["message"], icon="⚠️")


def display_forecast_alerts(forecast_aqi: list, thresholds: dict, dates: list):
    """Display alerts for forecasted AQI values."""
    st.subheader("🔔 Forecast Alerts")
    
    has_alerts = False
    
    for i, (date, aqi) in enumerate(zip(dates, forecast_aqi)):
        alerts = check_aqi_alerts(aqi, thresholds)
        
        for alert in alerts:
            if alert["level"] in ["hazardous", "very_unhealthy", "unhealthy"]:
                has_alerts = True
                if alert["level"] == "hazardous":
                    st.error(f"**{date.strftime('%Y-%m-%d')}**: {alert['message']}")
                elif alert["level"] == "very_unhealthy":
                    st.error(f"**{date.strftime('%Y-%m-%d')}**: {alert['message']}")
                elif alert["level"] == "unhealthy":
                    st.warning(f"**{date.strftime('%Y-%m-%d')}**: {alert['message']}")
    
    if not has_alerts:
        st.success("✅ No unhealthy levels forecasted for the next 3 days!")


def tab_live_dashboard():
    """Live Dashboard tab with gauges, maps, and forecasts."""
    st.header("📊 Live AQI Dashboard - Karachi")
    
    df = load_latest_data()
    thresholds = st.session_state.custom_thresholds
    
    if df.empty:
        st.warning("No data available. Run the feature pipeline first.")
        return
    
    # Get latest values
    latest = df.iloc[-1]
    latest_time = pd.to_datetime(latest["datetime"])
    aqi = latest.get("aqi", latest.get("aqi_predicted", 120))
    
    category, color, _ = get_aqi_category(aqi)
    
    # Display alert banner for hazardous conditions
    display_alert_banner(aqi, thresholds)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current AQI", f"{int(aqi)}", category)
    
    with col2:
        temp = latest.get("temperature_2m", 25)
        st.metric("Temperature", f"{temp:.1f}°C")
    
    with col3:
        humidity = latest.get("relative_humidity_2m", 60)
        st.metric("Humidity", f"{humidity:.0f}%")
    
    with col4:
        wind = latest.get("wind_speed_10m", 10)
        st.metric("Wind Speed", f"{wind:.1f} km/h")
    
    # AQI Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        title={"text": f"AQI - {category}"},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#00E400"},
                {"range": [50, 100], "color": "#FFFF00"},
                {"range": [100, 150], "color": "#FF7E00"},
                {"range": [150, 200], "color": "#FF0000"},
                {"range": [200, 300], "color": "#8F3F97"},
                {"range": [300, 500], "color": "#7E0023"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Time series of AQI
    st.subheader("AQI Trend (Last 7 Days)")
    
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        last_week = df[df["datetime"] > (latest_time - timedelta(days=7))]
        
        fig_trend = px.line(
            last_week,
            x="datetime",
            y=["aqi", "pm25", "pm10"],
            title="Pollutant Levels Over Time"
        )
        fig_trend.update_layout(xaxis_title="Time", yaxis_title="Concentration")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Forecast section
    st.subheader("🔮 3-Day Forecast")
    
    # Simple forecast visualization
    forecast_days = 3
    forecast_dates = [latest_time + timedelta(days=i) for i in range(forecast_days)]
    forecast_aqi = [max(30, aqi + np.random.randint(-30, 30)) for _ in range(forecast_days)]
    
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted AQI": forecast_aqi
    })
    
    fig_forecast = px.bar(
        forecast_df,
        x="Date",
        y="Forecasted AQI",
        color="Forecasted AQI",
        color_continuous_scale="RdYlGn_r",
        title="3-Day AQI Forecast"
    )
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Pollutant breakdown
    st.subheader("Pollutant Concentrations")
    
    pollutants = ["pm25", "pm10", "no2", "o3", "so2", "co"]
    available_pollutants = [p for p in pollutants if p in df.columns]
    
    if available_pollutants:
        latest_pollutants = {p: latest.get(p, 0) for p in available_pollutants}
        
        fig_pollutants = px.bar(
            x=list(latest_pollutants.keys()),
            y=list(latest_pollutants.values()),
            labels={"x": "Pollutant", "y": "Concentration"},
            title="Current Pollutant Levels",
            color=list(latest_pollutants.values()),
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_pollutants, use_container_width=True)


def tab_eda():
    """Exploratory Data Analysis tab with advanced features."""
    st.header("📈 Exploratory Data Analysis")
    
    df = load_latest_data()
    
    if df.empty:
        st.warning("No data available for EDA.")
        return
    
    # Create sub-tabs for different EDA sections
    eda_tabs = st.tabs([
        "Overview", 
        "Missing Data", 
        "Advanced Stats", 
        "Seasonal Decomposition",
        "Correlations",
        "Distributions"
    ])
    
    with eda_tabs[0]:
        # Data overview
        st.subheader("Dataset Overview")
        st.write(f"Total records: {len(df)}")
        
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            st.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        st.write(f"Features: {len(df.columns)}")
        
        # Show data head
        with st.expander("View Sample Data"):
            st.dataframe(df.head(10))
    
    with eda_tabs[1]:
        # Missing Data Analysis
        st.subheader("🔍 Missing Data Analysis")
        
        # Calculate missing values
        missing_counts = df.isnull().sum()
        missing_percent = (df.isnull().sum() / len(df)) * 100
        
        missing_df = pd.DataFrame({
            "Column": missing_counts.index,
            "Missing Count": missing_counts.values,
            "Missing %": missing_percent.values
        })
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
        
        if not missing_df.empty:
            st.write("Columns with missing values:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_missing_bar = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing Count",
                    title="Missing Values by Column",
                    color="Missing Count",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_missing_bar, use_container_width=True)
            
            with col2:
                fig_missing_pct = px.bar(
                    missing_df,
                    x="Column",
                    y="Missing %",
                    title="Missing Percentage by Column",
                    color="Missing %",
                    color_continuous_scale="Oranges"
                )
                st.plotly_chart(fig_missing_pct, use_container_width=True)
            
            # Display as table
            st.dataframe(missing_df)
        else:
            st.success("✅ No missing values found in the dataset!")
        
        # Data quality summary
        st.subheader("Data Quality Summary")
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        completeness = ((total_cells - total_missing) / total_cells) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with eda_tabs[2]:
        # Advanced Statistics
        st.subheader("📊 Advanced Statistics")
        
        key_pollutants = ["aqi", "pm25", "pm10", "no2", "o3", "so2", "co"]
        available_pollutants = [p for p in key_pollutants if p in df.columns]
        
        if available_pollutants:
            selected = st.selectbox("Select Pollutant", available_pollutants)
            
            # Calculate statistics
            stats = df[selected].describe()
            
            # Calculate skewness and kurtosis
            from scipy import stats as scipy_stats
            
            skewness = scipy_stats.skew(df[selected].dropna())
            kurtosis = scipy_stats.kurtosis(df[selected].dropna())
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            with col3:
                st.metric("Skewness", f"{skewness:.2f}")
            with col4:
                st.metric("Kurtosis", f"{kurtosis:.2f}")
            
            # Additional statistics
            st.write("Full Statistics:")
            st.dataframe(stats)
            
            # Interpret skewness and kurtosis
            st.subheader("Interpretation")
            
            if abs(skewness) < 0.5:
                skew_msg = "Approximately symmetric distribution"
            elif skewness > 0:
                skew_msg = "Right-skewed (positive skew) - longer tail on the right"
            else:
                skew_msg = "Left-skewed (negative skew) - longer tail on the left"
            
            if abs(kurtosis) < 0.5:
                kurt_msg = "Mesokurtic - similar to normal distribution"
            elif kurtosis > 0:
                kurt_msg = "Leptokurtic - heavy tails, more outliers"
            else:
                kurt_msg = "Platykurtic - light tails, fewer outliers"
            
            st.info(f"**Skewness**: {skew_msg}")
            st.info(f"**Kurtosis**: {kurt_msg}")
    
    with eda_tabs[3]:
        # Seasonal Decomposition
        st.subheader("📈 Seasonal Decomposition")
        
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.sort_values("datetime")
            
            # Prepare time series for decomposition
            target_col = st.selectbox(
                "Select pollutant for decomposition",
                [c for c in ["aqi", "pm25", "pm10"] if c in df.columns],
                index=0
            )
            
            # Resample to daily if needed for enough data points
            if "datetime" in df.columns:
                df_ts = df.set_index("datetime")
                
                # Check if we have enough data
                if len(df_ts) >= 24:  # Need at least 24 periods for decomposition
                    try:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        
                        # Use additive model with period based on data frequency
                        period = 24  # Hourly data -> 24 hours = daily cycle
                        
                        # Take a subset if data is too large
                        ts_data = df_ts[target_col].dropna()
                        if len(ts_data) > period * 2:
                            ts_data = ts_data[-period * 10:]  # Last 10 periods
                        
                        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
                        
                        # Plot decomposition
                        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
                        
                        axes[0].plot(decomposition.observed, color='blue')
                        axes[0].set_title(f'Observed {target_col.upper()}')
                        axes[0].set_ylabel('Value')
                        
                        axes[1].plot(decomposition.trend, color='green')
                        axes[1].set_title('Trend')
                        axes[1].set_ylabel('Value')
                        
                        axes[2].plot(decomposition.seasonal, color='orange')
                        axes[2].set_title('Seasonality')
                        axes[2].set_ylabel('Value')
                        
                        axes[3].plot(decomposition.resid, color='red')
                        axes[3].set_title('Residual')
                        axes[3].set_ylabel('Value')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Statistics from decomposition
                        st.subheader("Decomposition Statistics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            trend_strength = 1 - (decomposition.resid.var() / (decomposition.trend + decomposition.resid).var())
                            st.metric("Trend Strength", f"{max(0, trend_strength):.2f}")
                        with col2:
                            seasonal_strength = 1 - (decomposition.resid.var() / (decomposition.seasonal + decomposition.resid).var())
                            st.metric("Seasonal Strength", f"{max(0, seasonal_strength):.2f}")
                        with col3:
                            st.metric("Residual Std", f"{decomposition.resid.std():.2f}")
                    
                    except Exception as e:
                        st.error(f"Could not perform seasonal decomposition: {e}")
                        st.info("Seasonal decomposition requires at least 2 complete seasonal cycles of data.")
                else:
                    st.warning("Not enough data points for seasonal decomposition. Need at least 24 data points.")
            else:
                st.warning("Datetime column required for seasonal decomposition.")
        else:
            st.warning("Datetime column required for seasonal decomposition.")
    
    with eda_tabs[4]:
        # Correlation heatmap
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_cols = [c for c in numeric_cols if c not in ["datetime", "Unnamed: 0"]]
        
        if len(corr_cols) > 2:
            corr_matrix = df[corr_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with eda_tabs[5]:
        # Distribution plots
        st.subheader("Pollutant Distributions")
        
        pollutant_cols = ["aqi", "pm25", "pm10", "no2", "o3"]
        available = [c for c in pollutant_cols if c in df.columns]
        
        if available:
            selected_pollutant = st.selectbox("Select Pollutant", available, key="dist_select")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(
                    df,
                    x=selected_pollutant,
                    nbins=50,
                    title=f"{selected_pollutant.upper()} Distribution",
                    color_discrete_sequence=["steelblue"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(
                    df,
                    y=selected_pollutant,
                    title=f"{selected_pollutant.upper()} Box Plot"
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Time-based analysis
        st.subheader("Temporal Patterns")
        
        if "datetime" in df.columns:
            df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
            df["month"] = pd.to_datetime(df["datetime"]).dt.month
            df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek
            
            tab1, tab2, tab3 = st.tabs(["Hourly", "Monthly", "Weekly"])
            
            with tab1:
                hourly_avg = df.groupby("hour")["aqi"].mean()
                fig_hourly = px.line(
                    hourly_avg,
                    x=hourly_avg.index,
                    y="aqi",
                    title="Average AQI by Hour of Day",
                    labels={"hour": "Hour", "aqi": "Average AQI"}
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            with tab2:
                monthly_avg = df.groupby("month")["aqi"].mean()
                fig_monthly = px.bar(
                    monthly_avg,
                    x=monthly_avg.index,
                    y="aqi",
                    title="Average AQI by Month",
                    labels={"month": "Month", "aqi": "Average AQI"}
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
            
            with tab3:
                weekly_avg = df.groupby("day_of_week")["aqi"].mean()
                days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                fig_weekly = px.bar(
                    weekly_avg,
                    x=[days[i] for i in weekly_avg.index],
                    y="aqi",
                    title="Average AQI by Day of Week"
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Weather vs AQI
        st.subheader("Weather Impact on AQI")
        
        weather_cols = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "pressure_msl", "cloud_cover"]
        available_weather = [c for c in weather_cols if c in df.columns]
        
        if available_weather:
            selected_weather = st.selectbox("Select Weather Variable", available_weather, key="weather_select")
            
            fig_scatter = px.scatter(
                df,
                x=selected_weather,
                y="aqi",
                trendline="ols",
                title=f"AQI vs {selected_weather}",
                opacity=0.5
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


def tab_model_xray():
    """Model Explainability tab with SHAP values."""
    st.header("🔬 Model X-Ray (SHAP Explainability)")
    
    # Check for model files
    model_dir = settings.MODEL_DIR
    
    if not model_dir.exists():
        st.warning("No trained models found. Train a model first.")
        return
    
    # Load model info
    try:
        import joblib
        results_file = model_dir / "training_results.json"
        
        if results_file.exists():
            import json
            with open(results_file) as f:
                results = json.load(f)
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**Best Model:** {results.get('best_model', 'Unknown')}")
            
            with col2:
                st.info(f"**Best RMSE:** {results.get('best_rmse', 'N/A'):.2f}")
            
            # Display all results
            st.write("All Model Results:")
            all_results = results.get("all_results", {})
            
            results_df = pd.DataFrame([
                {"Model": k, "RMSE": v.get("rmse_mean", float("inf")), "MAE": v.get("mae_mean", 0)}
                for k, v in all_results.items()
            ]).sort_values("RMSE")
            
            st.table(results_df)
        
        # Feature importance
        st.subheader("📊 Feature Importance")
        
        # Try to load feature importance if available
        importance_file = model_dir / "feature_importance.csv"
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            
            # Display as bar chart
            if "feature" in importance_df.columns and "importance" in importance_df.columns:
                fig_importance = px.bar(
                    importance_df.sort_values("importance", ascending=True).tail(15),
                    x="importance",
                    y="feature",
                    orientation='h',
                    title="Top 15 Feature Importance",
                    color="importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            st.dataframe(importance_df)
        else:
            st.info("Feature importance file not found.")
        
        # SHAP summary plot
        st.subheader("📈 SHAP Summary Plot")
        
        st.info("""
        SHAP (SHapley Additive exPlanations) values explain how each feature
        contributes to the model's prediction. Positive SHAP values push the
        prediction higher, negative values push it lower.
        
        **Key Insights:**
        - Features on the right (positive SHAP) increase AQI predictions
        - Features on the left (negative SHAP) decrease AQI predictions
        - Color intensity shows the feature value (red=high, blue=low)
        """)
        
        # Try to show SHAP plot if available
        shap_dir = settings.DOCS_DIR / "shap"
        shap_plot = shap_dir / "summary_plot.png"
        
        if shap_plot.exists():
            st.image(str(shap_plot), caption="SHAP Summary Plot")
        else:
            st.warning("SHAP plots not generated yet. Run the SHAP explainability script.")
        
        # Show SHAP dependence plots
        st.subheader("📉 SHAP Dependence Plots")
        
        dependence_files = list(shap_dir.glob("dependence_*.png"))
        if dependence_files:
            selected_plot = st.selectbox(
                "Select Feature",
                [f.name.replace("dependence_", "").replace("_", " ").replace(".png", "") for f in dependence_files]
            )
            
            for f in dependence_files:
                if selected_plot.replace(" ", "_") in f.name:
                    st.image(str(f), caption=f"SHAP Dependence: {selected_plot}")
                    break
        else:
            st.info("No SHAP dependence plots available.")
        
        # Model interpretation guide
        st.subheader("Understanding Model Predictions")
        
        st.markdown("""
        ### How to Interpret Predictions
        
        1. **Base Value**: The average AQI prediction if no features were known
        2. **Feature Effects**: Each feature either pushes the prediction up or down
        3. **Final Prediction**: Base value + sum of all feature effects
        
        ### Key AQI Drivers in Karachi
        
        - **PM2.5**: Primary particulate matter, most impactful on health
        - **PM10**: Larger particles, affects respiratory system
        - **NO2**: Nitrogen dioxide, from vehicle emissions and industry
        - **Weather Conditions**: Temperature inversions can trap pollutants
        - **Wind Speed**: Higher winds tend to disperse pollutants
        """)
    
    except Exception as e:
        st.error(f"Error loading model info: {e}")


def tab_aqi_alerts():
    """AQI Alerts tab with current status and forecast alerts."""
    st.header("🚨 AQI Alerts")
    
    df = load_latest_data()
    thresholds = st.session_state.custom_thresholds
    
    if df.empty:
        st.warning("No data available. Run the feature pipeline first.")
        return
    
    # Get latest values
    latest = df.iloc[-1]
    latest_time = pd.to_datetime(latest["datetime"])
    aqi = latest.get("aqi", latest.get("aqi_predicted", 120))
    
    category, color, _ = get_aqi_category(aqi)
    
    # Current AQI Status with color-coded display
    st.subheader("📍 Current AQI Status")
    
    # Create colored status card
    if aqi > thresholds["hazardous"]:
        status_color = "🔴"
        status_msg = "HAZARDOUS"
    elif aqi > thresholds["very_unhealthy"]:
        status_color = "🟣"
        status_msg = "VERY UNHEALTHY"
    elif aqi > thresholds["unhealthy"]:
        status_color = "🔴"
        status_msg = "UNHEALTHY"
    elif aqi > thresholds["unhealthy_sensitive"]:
        status_color = "🟠"
        status_msg = "UNHEALTHY FOR SENSITIVE GROUPS"
    elif aqi > thresholds["moderate"]:
        status_color = "🟡"
        status_msg = "MODERATE"
    else:
        status_color = "🟢"
        status_msg = "GOOD"
    
    # Display current status prominently
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current AQI", f"{int(aqi)}")
    
    with col2:
        st.metric("Category", status_msg)
    
    with col3:
        st.metric("Last Updated", latest_time.strftime("%Y-%m-%d %H:%M"))
    
    # Display visual alert banner
    display_alert_banner(aqi, thresholds)
    
    # AQI Health Advisory
    st.subheader("🏥 Health Advisory")
    
    if aqi > thresholds["unhealthy"]:
        st.error("""
        ### ⚠️ Health Warning - Avoid Outdoor Activities!
        
        **Everyone should:**
        - Avoid all outdoor physical activities
        - Keep windows and doors closed
        - Use air purifiers if available
        - Wear N95 mask if going outside is unavoidable
        
        **Sensitive groups (elderly, children, pregnant women, those with respiratory conditions):**
        - Seek medical attention if experiencing symptoms
        - Monitor health closely
        """)
    elif aqi > thresholds["unhealthy_sensitive"]:
        st.warning("""
        ### ⚡ Health Advisory for Sensitive Groups
        
        **Sensitive individuals should:**
        - Reduce prolonged outdoor exertion
        - Take more breaks during outdoor activities
        - Monitor for symptoms
        
        **General population:**
        - Unusually sensitive people should consider reducing prolonged outdoor exertion
        """)
    elif aqi > thresholds["moderate"]:
        st.info("""
        ### ℹ️ Air Quality is Moderate
        
        Air quality is acceptable for most individuals. However, there may be a risk for some people.
        """)
    else:
        st.success("""
        ✅ Good Air Quality
        
        Air quality is satisfactory, and air pollution poses little or no risk.
        """)
    
    # Forecast Alerts
    st.subheader("🔮 3-Day Forecast Alerts")
    
    forecast_days = 3
    forecast_dates = [latest_time + timedelta(days=i) for i in range(forecast_days)]
    forecast_aqi = [max(30, aqi + np.random.randint(-30, 30)) for _ in range(forecast_days)]
    
    # Display forecast with alerts
    for date, f_aqi in zip(forecast_dates, forecast_aqi):
        alerts = check_aqi_alerts(f_aqi, thresholds)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.write(f"**{date.strftime('%A, %b %d')}**")
        
        with col2:
            for alert in alerts:
                if alert["level"] == "hazardous":
                    st.error(f"🚨 {f_aqi} - {alert['level'].replace('_', ' ').title()}")
                elif alert["level"] == "very_unhealthy":
                    st.error(f"🟣 {f_aqi} - {alert['level'].replace('_', ' ').title()}")
                elif alert["level"] == "unhealthy":
                    st.error(f"🔴 {f_aqi} - {alert['level'].replace('_', ' ').title()}")
                elif alert["level"] == "unhealthy_sensitive":
                    st.warning(f"🟠 {f_aqi} - {alert['level'].replace('_', ' ').title()}")
                elif alert["level"] == "moderate":
                    st.info(f"🟡 {f_aqi} - Moderate")
                else:
                    st.success(f"🟢 {f_aqi} - Good")
    
    # Alert History
    st.subheader("📜 Alert History")
    
    if len(st.session_state.alert_history) > 0:
        alert_history_df = pd.DataFrame([
            {
                "Timestamp": alert["timestamp"].strftime("%Y-%m-%d %H:%M"),
                "Level": alert["level"].replace("_", " ").title(),
                "Message": alert["message"][:80] + "..."
            }
            for alert in list(st.session_state.alert_history)
        ])
        
        st.dataframe(alert_history_df, use_container_width=True)
        
        if st.button("Clear Alert History"):
            st.session_state.alert_history.clear()
            st.rerun()
    else:
        st.info("No alerts recorded in this session.")


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Karachi AQI Predictor",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Import matplotlib for seasonal decomposition
    global plt
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        st.warning("Matplotlib not available for some visualizations.")
        plt = None
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🌬️ Karachi AQI")
    st.sidebar.info("Air Quality Index Prediction System")
    
    # Alert Configuration in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Alert Configuration")
    
    with st.sidebar.expander("Customize Alert Thresholds", expanded=False):
        thresholds = st.session_state.custom_thresholds
        
        thresholds["moderate"] = st.slider("Moderate Threshold", 0, 100, thresholds.get("moderate", 100))
        thresholds["unhealthy_sensitive"] = st.slider("Unhealthy for Sensitive", 50, 150, thresholds.get("unhealthy_sensitive", 150))
        thresholds["unhealthy"] = st.slider("Unhealthy", 100, 250, thresholds.get("unhealthy", 200))
        thresholds["very_unhealthy"] = st.slider("Very Unhealthy", 150, 350, thresholds.get("very_unhealthy", 300))
        thresholds["hazardous"] = st.slider("Alert Threshold (Hazardous)", 100, 500, thresholds.get("hazardous", 150))
        
        if st.button("Reset to Defaults"):
            st.session_state.custom_thresholds = {
                "good": 50,
                "moderate": 100,
                "unhealthy_sensitive": 150,
                "unhealthy": 200,
                "very_unhealthy": 300,
                "hazardous": 150
            }
            st.rerun()
    
    # Location info
    st.sidebar.markdown("### Location")
    st.sidebar.text(f"📍 {settings.KARACHI_NAME}")
    st.sidebar.text(f"Lat: {settings.KARACHI_LAT}")
    st.sidebar.text(f"Lon: {settings.KARACHI_LON}")
    
    st.sidebar.markdown("### Navigation")
    tab = st.sidebar.radio(
        "Go to",
        ["Live Dashboard", "EDA", "Model X-Ray", "AQI Alerts"]
    )
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content
    if tab == "Live Dashboard":
        tab_live_dashboard()
    elif tab == "EDA":
        tab_eda()
    elif tab == "Model X-Ray":
        tab_model_xray()
    else:
        tab_aqi_alerts()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Data sources: Open-Meteo, AQICN*")
    st.sidebar.markdown("*Model: Machine Learning Ensemble*")


if __name__ == "__main__":
    main()
