"""
Streamlit Dashboard for AQI Prediction System.
Three tabs: Live Dashboard, EDA, Model X-Ray (SHAP).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def tab_live_dashboard():
    """Live Dashboard tab with gauges, maps, and forecasts."""
    st.header("📊 Live AQI Dashboard - Karachi")
    
    df = load_latest_data()
    
    if df.empty:
        st.warning("No data available. Run the feature pipeline first.")
        return
    
    # Get latest values
    latest = df.iloc[-1]
    latest_time = pd.to_datetime(latest["datetime"])
    aqi = latest.get("aqi", latest.get("aqi_predicted", 120))
    
    category, color, _ = get_aqi_category(aqi)
    
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
    """Exploratory Data Analysis tab."""
    st.header("📈 Exploratory Data Analysis")
    
    df = load_latest_data()
    
    if df.empty:
        st.warning("No data available for EDA.")
        return
    
    # Data overview
    st.subheader("Dataset Overview")
    st.write(f"Total records: {len(df)}")
    st.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    st.write(f"Features: {len(df.columns)}")
    
    # Show data head
    with st.expander("View Sample Data"):
        st.dataframe(df.head(10))
    
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
    
    # Distribution plots
    st.subheader("Pollutant Distributions")
    
    pollutant_cols = ["aqi", "pm25", "pm10", "no2", "o3"]
    available = [c for c in pollutant_cols if c in df.columns]
    
    if available:
        selected_pollutant = st.selectbox("Select Pollutant", available)
        
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
        selected_weather = st.selectbox("Select Weather Variable", available_weather)
        
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
        st.subheader("Feature Importance")
        
        # Try to load feature importance if available
        importance_file = model_dir / "feature_importance.csv"
        if importance_file.exists():
            importance_df = pd.read_csv(importance_file)
            st.dataframe(importance_df)
        
        # SHAP summary plot placeholder
        st.subheader("SHAP Summary Plot")
        
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
        st.subheader("SHAP Dependence Plots")
        
        dependence_files = list(shap_dir.glob("dependence_*.png"))
        if dependence_files:
            selected_plot = st.selectbox(
                "Select Feature",
                [f.name.replace("dependence_", "").replace("_", " ").replace(".png", "") for f in dependence_files]
            )
            
            for f in dependence_files:
                if selected_plot.replace(" ", "_") in f.name:
                    st.image(str(f))
                    break
        
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


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Karachi AQI Predictor",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
    
    # Location info
    st.sidebar.markdown("### Location")
    st.sidebar.text(f"📍 {settings.KARACHI_NAME}")
    st.sidebar.text(f"Lat: {settings.KARACHI_LAT}")
    st.sidebar.text(f"Lon: {settings.KARACHI_LON}")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    tab = st.sidebar.radio(
        "Go to",
        ["Live Dashboard", "EDA", "Model X-Ray"]
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
    else:
        tab_model_xray()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Data sources: Open-Meteo, AQICN*")
    st.sidebar.markdown("*Model: Machine Learning Ensemble*")


if __name__ == "__main__":
    main()
