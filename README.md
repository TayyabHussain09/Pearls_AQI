# Karachi AQI Prediction System

A production-ready, serverless Air Quality Index (AQI) Prediction System for Karachi, Pakistan. This project implements an end-to-end machine learning pipeline with automated data collection, feature engineering, model training, and real-time predictions through an interactive web dashboard.

## Project Overview

This system predicts the Air Quality Index (AQI) in Karachi, Pakistan for the next 3 days using weather and pollutant data. It uses a 100% serverless stack with Hopsworks Feature Store and Model Registry integration.

## Features

### 1. Data Collection
- Fetches historical weather data from Open-Meteo Archive API (free, up to 40+ years)
- Fetches current AQI data from AQICN, OpenWeatherMap, and WeatherAPI
- Automatic data backfilling for 2+ years of historical data
- Hourly incremental updates for real-time data

### 2. Feature Engineering
- **Cyclical Time Encodings**: Sin/Cos transformations for hour, day of week, month, and day of year
- **Rolling Window Features**: 24-hour and 7-day rolling averages, min, max, std
- **Lag Features**: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, and 7-day lags
- **Weather Interactions**: Temperature-humidity, wind-pressure, visibility-humidity
- **Karachi-Specific Features**: Sea breeze effect, temperature inversion risk, pollution accumulation risk, seasonal indicators (monsoon, winter, summer)

### 3. Model Training
- **11 ML Models** trained and evaluated:
  1. Ridge Regression (Best: RMSE=0.0144)
  2. Extra Trees (RMSE=0.0449)
  3. Gradient Boosting (RMSE=0.0522)
  4. Random Forest (RMSE=0.0598)
  5. XGBoost (RMSE=0.0654)
  6. Lasso (RMSE=0.0953)
  7. ElasticNet (RMSE=0.1542)
  8. LightGBM (RMSE=0.1234)
  9. AdaBoost (RMSE=0.9841)
  10. KNN (RMSE=2.3656)
  11. TensorFlow Neural Network (RMSE=16.01)
- Time-series cross-validation (5 folds)
- Automatic model selection based on RMSE
- Model registry for version control

### 4. Web Dashboard
- **Live Dashboard**: Real-time AQI gauges, trend charts, 3-day forecast
- **EDA Tab**: Correlation heatmaps, pollutant distributions, temporal patterns
- **Model X-Ray**: SHAP feature importance, model interpretation

### 5. Automation
- **GitHub Actions** workflows:
  - Feature pipeline: Runs hourly
  - Training pipeline: Runs daily
- **Hopsworks Integration**: Feature Store and Model Registry

## Project Structure

```
aqi_predictor/
├── api/                          # API integration module
│   ├── fetchers/
│   │   ├── base.py              # Base fetcher class
│   │   └── providers.py          # Open-Meteo, AQICN, OpenWeatherMap, WeatherAPI
│   ├── schemas/
│   │   ├── data_models.py        # Pydantic data models
│   │   └── models.py             # Response models
│   └── orchestrator.py           # Data orchestration
├── config/
│   └── settings.py               # Configuration with Karachi coordinates
├── dashboard/
│   └── app.py                   # Streamlit dashboard with 3 tabs
├── features/
│   ├── feature_engineering.py    # CyclicalTimeEncoder, RollingFeatureGenerator, LagFeatureGenerator
│   └── main.py                  # Feature pipeline entry point
├── models/
│   ├── trainer.py                # ModelFactory with 11 models
│   ├── training_pipeline.py      # Training orchestration
│   └── registry/
│       ├── hopsworks_registry.py  # Hopsworks integration
│       └── hopsworks_pipeline.py  # Complete Hopsworks pipeline
├── data/
│   ├── raw/                      # Raw data storage
│   └── processed/                # Processed features
├── models/karachi/               # Trained models
├── .github/workflows/           # GitHub Actions
│   ├── feature-pipeline.yml     # Hourly feature pipeline
│   └── training-pipeline.yml    # Daily training
├── requirements.txt             # Dependencies
└── run_pipeline.py              # Main entry point
```

## Installation

1. **Clone the repository**
```bash
cd c:/Users/tayya/Desktop/aqi_predictor
```

2. **Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Copy .env.example to .env and fill in your values
copy .env.example .env
```

Required environment variables:
- `HOPSWORKS_API_KEY`: Your Hopsworks API key (for Feature Store & Model Registry)
- `HOPSWORKS_PROJECT_NAME`: Your Hopsworks project name
- `AQICN_API_TOKEN`: AQICN API token (get from https://aqicn.org/data-platform/token/)
- `OPENWEATHERMAP_API_KEY`: OpenWeatherMap API key
- `WEATHERAPI_KEY`: WeatherAPI key

## Usage

### Run Feature Pipeline
```bash
# Backfill 2 years of historical data
python run_pipeline.py --mode features --feature-mode backfill --years 2

# Incremental update (last 24 hours)
python run_pipeline.py --mode features --feature-mode incremental
```

### Train Models
```bash
python run_pipeline.py --mode train
```

### Run Full Pipeline
```bash
# With Hopsworks storage
python run_pipeline.py --mode full --hopsworks

# Without Hopsworks (local only)
python run_pipeline.py --mode full
```

### Start Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will be available at `http://localhost:8501`

## Data Summary

- **Location**: Karachi, Pakistan (24.8607° N, 67.0011° E)
- **Training Data**: 18,120 records (Feb 2024 - Feb 2026)
- **Features**: 127 columns including weather, pollutants, time encodings, rolling features, and lag features
- **Target**: AQI (Air Quality Index)
- **Best Model**: Ridge Regression (RMSE=0.0144, R²=0.9999)

## AQI Categories

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0-50 | Good | Green |
| 51-100 | Moderate | Yellow |
| 101-150 | Unhealthy for Sensitive Groups | Orange |
| 151-200 | Unhealthy | Red |
| 201-300 | Very Unhealthy | Purple |
| 301-500 | Hazardous | Maroon |

## Karachi-Specific Features

The system includes several Karachi-specific features:

1. **Sea Breeze Effect**: Wind from the Arabian Sea (West direction) affects air quality
2. **Temperature Inversion Risk**: Winter nights with high pressure and low clouds trap pollutants
3. **Pollution Accumulation Risk**: Low wind, high pressure, and no precipitation lead to pollution buildup
4. **Seasonal Indicators**: Monsoon (June-September), Winter (December-February), Summer (May-July)

## Model Performance

All models are evaluated using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

Results are stored in `models/karachi/training_results.json`.

## Automation

### GitHub Actions

The project includes two GitHub Actions workflows:

1. **feature-pipeline.yml**: Runs every hour to fetch latest data and update features
2. **training-pipeline.yml**: Runs daily to retrain models with new data

To enable GitHub Actions:
1. Push your code to GitHub
2. Go to Actions tab in your repository
3. The workflows will run automatically

### Hopsworks Integration

The system integrates with Hopsworks for:
- **Feature Store**: Store and retrieve engineered features
- **Model Registry**: Version control for trained models

To use Hopsworks:
1. Create a free account at https://www.hopsworks.ai/
2. Generate an API key
3. Add the API key to your .env file

## Dependencies

See `requirements.txt` for the complete list of dependencies:
- Core: numpy, pandas, scipy
- ML: scikit-learn, xgboost, lightgbm
- Deep Learning: tensorflow
- Explainability: shap
- Feature Store: hopsworks
- Web: streamlit, fastapi
- Visualization: plotly, matplotlib, seaborn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License

## Acknowledgments

- Open-Meteo for free weather data
- AQICN for AQI data
- Hopsworks for Feature Store and Model Registry
- Streamlit for the web dashboard
