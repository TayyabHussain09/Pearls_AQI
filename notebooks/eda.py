"""
Exploratory Data Analysis (EDA) for Karachi AQI Prediction
===========================================================
This script performs comprehensive EDA on the AQI dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def load_data(data_path="data/processed/"):
    """Load the processed feature data."""
    data_dir = Path(data_path)
    
    # Find the latest processed data file
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            print(f"Loading data from: {latest_file}")
            df = pd.read_csv(latest_file)
            return df
    
    # Try alternative paths
    for path in ["data/karachi_features.csv", "data/processed/karachi_features.csv"]:
        if Path(path).exists():
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            return df
    
    print("No data file found. Please run feature pipeline first.")
    return None


def basic_statistics(df):
    """Display basic statistics of the dataset."""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nDate Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Target variable statistics
    if 'aqi' in df.columns:
        print(f"\n--- AQI Statistics ---")
        print(f"Mean AQI: {df['aqi'].mean():.2f}")
        print(f"Median AQI: {df['aqi'].median():.2f}")
        print(f"Std AQI: {df['aqi'].std():.2f}")
        print(f"Min AQI: {df['aqi'].min():.2f}")
        print(f"Max AQI: {df['aqi'].max():.2f}")
        
        # AQI Categories
        print(f"\n--- AQI Category Distribution ---")
        aqi_categories = pd.cut(df['aqi'], 
                               bins=[0, 50, 100, 150, 200, 300, 500],
                               labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 
                                      'Unhealthy', 'Very_Unhealthy', 'Hazardous'])
        print(aqi_categories.value_counts().sort_index())
    
    # Missing values
    print(f"\n--- Missing Values ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
    print(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False).head(10))


def temporal_analysis(df):
    """Analyze temporal patterns in AQI."""
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # Monthly trends
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    print("\n--- Monthly Average AQI ---")
    monthly = df.groupby('month')['aqi'].agg(['mean', 'std', 'median'])
    print(monthly.round(2))
    
    print("\n--- Hourly Average AQI ---")
    hourly = df.groupby('hour')['aqi'].agg(['mean', 'std', 'median'])
    print(hourly.round(2))
    
    print("\n--- Day of Week Average AQI ---")
    dow = df.groupby('day_of_week')['aqi'].agg(['mean', 'std', 'median'])
    dow.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(dow.round(2))
    
    # Yearly comparison
    print("\n--- Yearly Average AQI ---")
    yearly = df.groupby('year')['aqi'].agg(['mean', 'std', 'count'])
    print(yearly.round(2))


def pollutant_analysis(df):
    """Analyze relationships between pollutants and AQI."""
    print("\n" + "="*60)
    print("POLLUTANT ANALYSIS")
    print("="*60)
    
    # Identify pollutant columns
    pollutant_cols = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
    available_pollutants = [col for col in pollutant_cols if col in df.columns]
    
    print(f"\nAvailable pollutants: {available_pollutants}")
    
    if available_pollutants:
        print("\n--- Correlation with AQI ---")
        correlations = df[available_pollutants + ['aqi']].corr()['aqi'].drop('aqi').sort_values(ascending=False)
        print(correlations.round(3))
        
        print("\n--- Pollutant Statistics ---")
        for col in available_pollutants:
            print(f"\n{col.upper()}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std: {df[col].std():.2f}")
            print(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")


def weather_analysis(df):
    """Analyze weather features and their relationship with AQI."""
    print("\n" + "="*60)
    print("WEATHER FEATURE ANALYSIS")
    print("="*60)
    
    # Identify weather columns
    weather_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                   'precipitation', 'cloud_cover', 'visibility']
    available_weather = [col for col in weather_cols if col in df.columns]
    
    print(f"\nAvailable weather features: {available_weather}")
    
    if available_weather:
        print("\n--- Correlation with AQI ---")
        correlations = df[available_weather + ['aqi']].corr()['aqi'].drop('aqi').sort_values(key=abs, ascending=False)
        print(correlations.round(3))


def feature_importance_preview(df):
    """Preview potential important features."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE PREVIEW")
    print("="*60)
    
    # Get feature columns (excluding target and metadata)
    exclude_cols = ['aqi', 'datetime', 'city', 'aqi_category']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\nTotal features available: {len(feature_cols)}")
    
    # Show lag and rolling features
    lag_features = [col for col in feature_cols if 'lag' in col.lower()]
    rolling_features = [col for col in feature_cols if 'rolling' in col.lower()]
    cyclical_features = [col for col in feature_cols if 'sin' in col.lower() or 'cos' in col.lower()]
    
    print(f"Lag features: {len(lag_features)}")
    print(f"Rolling features: {len(rolling_features)}")
    print(f"Cyclical features: {len(cyclical_features)}")
    
    # Show sample of most variable features
    if len(feature_cols) > 0:
        variances = df[feature_cols].var().sort_values(ascending=False).head(10)
        print("\nTop 10 most variable features:")
        print(variances.round(2))


def generate_visualizations(df, output_dir="notebooks/figures/"):
    """Generate and save EDA visualizations."""
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')
    
    # 1. AQI Time Series
    plt.figure(figsize=(14, 5))
    plt.plot(df['datetime'], df['aqi'], alpha=0.5, linewidth=0.5)
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Karachi AQI Time Series')
    plt.tight_layout()
    plt.savefig(output_path / 'aqi_timeseries.png', dpi=150)
    plt.close()
    print("Saved: aqi_timeseries.png")
    
    # 2. AQI Distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['aqi'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('AQI')
    plt.ylabel('Frequency')
    plt.title('AQI Distribution')
    
    plt.subplot(1, 2, 2)
    df['aqi'].dropna().plot(kind='box')
    plt.title('AQI Box Plot')
    plt.tight_layout()
    plt.savefig(output_path / 'aqi_distribution.png', dpi=150)
    plt.close()
    print("Saved: aqi_distribution.png")
    
    # 3. Monthly AQI Pattern
    plt.figure(figsize=(10, 5))
    monthly = df.groupby(df['datetime'].dt.month)['aqi'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(range(1, 13), monthly.values, color='steelblue', alpha=0.7)
    plt.xticks(range(1, 13), months)
    plt.xlabel('Month')
    plt.ylabel('Average AQI')
    plt.title('Monthly Average AQI Pattern - Karachi')
    plt.tight_layout()
    plt.savefig(output_path / 'monthly_aqi.png', dpi=150)
    plt.close()
    print("Saved: monthly_aqi.png")
    
    # 4. Hourly AQI Pattern
    plt.figure(figsize=(10, 5))
    hourly = df.groupby(df['datetime'].dt.hour)['aqi'].mean()
    plt.plot(hourly.index, hourly.values, marker='o', linewidth=2, markersize=4)
    plt.xlabel('Hour of Day')
    plt.ylabel('Average AQI')
    plt.title('Hourly Average AQI Pattern - Karachi')
    plt.xticks(range(0, 24, 2))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'hourly_aqi.png', dpi=150)
    plt.close()
    print("Saved: hourly_aqi.png")
    
    # 5. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr_cols = ['aqi', 'pm25', 'pm10', 'temperature', 'humidity', 'wind_speed']
    available_corr = [col for col in corr_cols if col in df.columns]
    if len(available_corr) > 2:
        corr_matrix = df[available_corr].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=150)
        plt.close()
        print("Saved: correlation_heatmap.png")
    
    # 6. AQI Category Distribution
    plt.figure(figsize=(8, 5))
    aqi_categories = pd.cut(df['aqi'], 
                           bins=[0, 50, 100, 150, 200, 300, 500],
                           labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 
                                  'Unhealthy', 'Very_Unhealthy', 'Hazardous'])
    aqi_categories.value_counts().plot(kind='bar', color='steelblue', alpha=0.7)
    plt.xlabel('AQI Category')
    plt.ylabel('Count')
    plt.title('AQI Category Distribution - Karachi')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'aqi_categories.png', dpi=150)
    plt.close()
    print("Saved: aqi_categories.png")
    
    print(f"\nAll visualizations saved to: {output_path}")


def main():
    """Run complete EDA."""
    print("="*60)
    print("KARACHI AQI PREDICTION - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data()
    if df is None:
        print("Error: Could not load data. Please run the feature pipeline first.")
        return
    
    print(f"\nLoaded dataset with {len(df)} records and {len(df.columns)} features")
    
    # Run all analyses
    basic_statistics(df)
    temporal_analysis(df)
    pollutant_analysis(df)
    weather_analysis(df)
    feature_importance_preview(df)
    
    # Generate visualizations
    generate_visualizations(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nKey Insights for Karachi AQI:")
    print("1. AQI follows clear seasonal patterns")
    print("2. Winter months typically have higher AQI")
    print("3. Hourly patterns show morning and evening peaks")
    print("4. PM2.5 and PM10 are major contributors to AQI")
    print("5. Weather factors (wind, humidity) affect AQI dispersion")


if __name__ == "__main__":
    main()
