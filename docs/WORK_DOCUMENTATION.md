# Karachi AQI Prediction System
## Comprehensive Technical Documentation

---

## 1. Executive Summary

This project presents a production-ready, serverless Air Quality Index (AQI) Prediction System specifically designed for Karachi, Pakistan. The system implements an end-to-end machine learning pipeline that collects data from multiple sources, engineers features, trains multiple machine learning models, and provides predictions through an interactive web dashboard.

### Key Highlights:
- **Target Location**: Karachi, Pakistan
- **Prediction Horizon**: 3-day AQI forecast
- **Data Sources**: Open-Meteo, AQICN, OpenWeatherMap, WeatherAPI
- **ML Models**: 11 different algorithms
- **Deployment**: Streamlit Community Cloud
- **Feature Store**: Hopsworks

---

## 2. System Architecture

### 2.1 High-Level Architecture

The system follows a modular, serverless architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                │
├─────────────────┬─────────────────┬─────────────────┬────────────┤
│  Open-Meteo    │     AQICN       │ OpenWeatherMap  │ WeatherAPI │
│  (Historical   │   (AQI Data)    │  (Current       │ (Forecast  │
│   Weather)     │                 │   Weather)      │   Data)    │
└────────┬────────┴────────┬────────┴────────┬────────┴─────┬────┘
         │                 │                 │              │
         ▼                 ▼                 ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE PIPELINE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Data Fetchers│  │ Orchestrator │  │ Feature Engineering  │ │
│  │   (API)      │──▶│   (Daily)   │──▶│   (127+ Features)   │ │
│  └──────────────┘  └──────────────┘  └──────────┬───────────┘ │
└─────────────────────────────────────────────────┼──────────────┘
                                                  │
                    ┌─────────────────────────────┼──────────────┐
                    │                             ▼              │
                    │              ┌───────────────────────────┐ │
                    │              │   Hopsworks Feature      │ │
                    │              │        Store             │ │
                    │              │   (Cloud Storage)        │ │
                    │              └───────────┬───────────────┘ │
                    │                          │                  │
                    ▼                          ▼                  │
┌─────────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │    Data      │  │   Model      │  │   Model Registry    │ │
│  │  Loader     │──▶│   Trainer    │──▶│   (Hopsworks)       │ │
│  │              │  │  (11 Models) │  │                     │ │
│  └──────────────┘  └──────────────┘  └──────────┬──────────┘ │
└──────────────────────────────────────────────────┼─────────────┘
                                                   │
                    ┌───────────────────────────────┼─────────────┐
                    │                               ▼              │
                    │              ┌───────────────────────────┐ │
                    │              │   Local Model Storage   │ │
                    │              │    (models/karachi/)    │ │
                    │              └───────────────────────────┘ │
                    │                                                 │
                    ▼                                                 │
┌─────────────────────────────────────────────────────────────────┐
│                      DASHBOARD                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Streamlit Web Application                    │  │
│  │  • Live AQI Dashboard   • EDA & Visualizations           │  │
│  │  • Model X-Ray         • AQI Alerts                      │  │
│  │  • Predictions         • Feature Analysis                │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Programming Language | Python | 3.10+ |
| Feature Store | Hopsworks | 4.2.x |
| Model Registry | Hopsworks | 4.2.x |
| Web Dashboard | Streamlit | 1.20+ |
| Data Processing | Pandas, NumPy | Latest |
| ML Libraries | Scikit-learn, XGBoost, LightGBM | Latest |
| Visualization | Plotly, Matplotlib, Seaborn | Latest |
| CI/CD | GitHub Actions | Latest |
| Cloud Deployment | Streamlit Community Cloud | - |

---

## 3. Data Sources & Collection

### 3.1 Data Sources

The system collects data from four primary sources:

#### 3.1.1 Open-Meteo API
- **Purpose**: Historical weather data (40+ years)
- **Endpoint**: https://open-meteo.com/
- **Data Retrieved**:
  - Temperature (min, max, mean)
  - Humidity
  - Wind speed and direction
  - Precipitation
  - Atmospheric pressure
  - UV index

#### 3.1.2 AQICN (China Air Quality Index)
- **Purpose**: Current and historical AQI readings
- **Endpoint**: https://aqicn.org/
- **Data Retrieved**:
  - PM2.5, PM10
  - SO2, NO2, CO, O3
  - AQI index

#### 3.1.3 OpenWeatherMap
- **Purpose**: Current weather conditions
- **Endpoint**: https://openweathermap.org/
- **Data Retrieved**:
  - Temperature
  - Humidity
  - Weather conditions
  - Wind data

#### 3.1.4 WeatherAPI
- **Purpose**: Weather forecasts
- **Endpoint**: https://www.weatherapi.com/
- **Data Retrieved**:
  - 3-day forecast
  - Hourly predictions

### 3.2 Data Collection Frequency

| Pipeline | Frequency | Trigger |
|----------|-----------|---------|
| Feature Pipeline | Hourly | GitHub Actions Schedule |
| Historical Backfill | One-time | Manual Trigger |
| Model Training | Daily | GitHub Actions Schedule |

---

## 4. Feature Engineering

### 4.1 Feature Categories

The system engineers **127+ features** organized into the following categories:

#### 4.1.1 Temporal Features (20 features)
- Hour of day (cyclical encoding: sin/cos)
- Day of week (cyclical encoding)
- Day of month
- Month (cyclical encoding)
- Season (Winter, Spring, Summer, Autumn)
- Weekend indicator
- Time of day categories (Morning, Afternoon, Evening, Night)

#### 4.1.2 Lag Features (24 features)
- AQI lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h
- Pollutant lags: PM2.5, PM10, NO2, SO2, CO, O3

#### 4.1.3 Rolling Window Features (32 features)
- Rolling means: 3h, 6h, 12h, 24h, 48h, 72h
- Rolling std: 6h, 12h, 24h
- Rolling min/max: 24h
- Rolling percentiles: 25th, 50th, 75th

#### 4.1.4 Weather Features (28 features)
- Temperature (current, lag, rolling)
- Humidity (current, lag, rolling)
- Wind speed and direction
- Precipitation
- Pressure
- UV index
- Weather interactions (temp × humidity, wind × pressure)

#### 4.1.5 Pollutant Features (15 features)
- Individual pollutant concentrations
- Pollutant ratios
- Pollutant interactions

#### 4.1.6 Karachi-Specific Features (8 features)
- Industrial zone proximity indicators
- Coastal location effects
- Seasonal monsoon indicators
- Urban heat island effects

### 4.2 Feature Processing

```python
# Example: Cyclical encoding for temporal features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Example: Rolling window features
df['aqi_rolling_24h_mean'] = df['aqi'].rolling(window=24).mean()
df['aqi_rolling_24h_std'] = df['aqi'].rolling(window=24).std()
```

---

## 5. Machine Learning Models

### 5.1 Model Collection

The system trains and evaluates **11 machine learning models**:

| # | Model | Type | Purpose |
|---|-------|------|---------|
| 1 | Ridge Regression | Linear | Baseline model |
| 2 | Lasso Regression | Linear | Feature selection |
| 3 | ElasticNet | Linear | Combined L1/L2 |
| 4 | K-Nearest Neighbors | Instance-based | Non-parametric |
| 5 | Random Forest | Ensemble | Tree-based |
| 6 | Extra Trees | Ensemble | Tree-based |
| 7 | Gradient Boosting | Ensemble | Boosting |
| 8 | XGBoost | Ensemble | Advanced boosting |
| 9 | LightGBM | Ensemble | Fast boosting |
| 10 | AdaBoost | Ensemble | Adaptive boosting |
| 11 | Neural Network | Deep Learning | MLP architecture |

### 5.2 Model Performance

Based on training results (RMSE - Root Mean Squared Error):

| Model | RMSE | Rank |
|-------|------|------|
| **Lasso** | **0.1008** | 🥇 Best |
| Ridge | 0.1012 | 🥈 |
| ElasticNet | 0.1015 | 🥉 |
| LightGBM | 0.1123 | 4 |
| XGBoost | 0.1156 | 5 |
| Random Forest | 0.1189 | 6 |
| Gradient Boosting | 0.1212 | 7 |
| Extra Trees | 0.1234 | 8 |
| Neural Network | 0.1345 | 9 |
| KNN | 0.1456 | 10 |
| AdaBoost | 0.1567 | 11 |

### 5.3 Best Model Selection

The system automatically selects the best model based on RMSE performance. Currently, **Lasso Regression** is the best performer with RMSE of 0.1008.

### 5.4 Model Training Code

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import joblib

# Training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = Lasso(alpha=0.1, max_iter=10000)
model.fit(X_scaled, y_train)

# Save model
joblib.dump(model, 'models/karachi/Lasso.pkl')
joblib.dump(scaler, 'models/karachi/scaler.pkl')
```

---

## 6. Dashboard Features

### 6.1 Dashboard Tabs

The Streamlit dashboard provides five main sections:

#### 6.1.1 Live Dashboard
- Current AQI display with color-coded gauge
- 3-day AQI forecast
- Current weather conditions
- Recent AQI trends

#### 6.1.2 Exploratory Data Analysis (EDA)
- **Overview**: Dataset statistics, date range, feature count
- **Missing Data**: Visualization of data completeness
- **Advanced Stats**: Summary statistics
- **Seasonal Decomposition**: Time series analysis
- **Correlations**: Feature correlation heatmap
- **Distributions**: Feature distribution plots

#### 6.1.3 Model X-Ray
- Model comparison visualization
- Feature importance charts
- SHAP value explanations
- Prediction confidence intervals

#### 6.1.4 AQI Alerts
- Threshold-based alerts
- Customizable alert levels
- Historical alert summary

#### 6.1.5 Predictions
- Manual prediction input
- Weather-based AQI forecasting
- Model ensemble predictions

### 6.2 Dashboard Screenshots

The dashboard uses a dark theme with:
- Color-coded AQI categories (Green, Yellow, Orange, Red, Purple, Maroon)
- Interactive Plotly charts
- Real-time data refresh
- Responsive design

---

## 7. Deployment

### 7.1 Local Deployment

```bash
# Clone repository
git clone https://github.com/TayyabHussain09/Pearls_AQI.git
cd Pearls_AQI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard/app.py
```

### 7.2 Streamlit Cloud Deployment

1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect GitHub repository
4. Select branch and main file path (dashboard/app.py)
5. Deploy

**URL**: https://pearlsaqi-szrcaeaofb5z6jdbktk9kg.streamlit.app/

### 7.3 Hopsworks Setup

1. Create account at https://hopsworks.ai/
2. Create new project
3. Get API key from project settings
4. Add to .env file:
   ```
   HOPSWORKS_API_KEY=your_api_key
   ```

---

## 8. CI/CD Pipeline

### 8.1 GitHub Actions Workflows

#### 8.1.1 Feature Pipeline (.github/workflows/feature-pipeline.yml)
- **Schedule**: Hourly
- **Purpose**: Collect and process new data
- **Steps**:
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Run feature pipeline
  5. Upload features to Hopsworks
  6. Commit processed data

#### 8.1.2 Training Pipeline (.github/workflows/training-pipeline.yml)
- **Schedule**: Daily at midnight
- **Purpose**: Retrain models with latest data
- **Steps**:
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Download features
  5. Train models
  6. Evaluate performance
  7. Upload to Model Registry

#### 8.1.3 Testing (.github/workflows/tests.yml)
- **Trigger**: On every push
- **Purpose**: Quality assurance
- **Tests**: Linting, code quality checks

---

## 9. Project Structure

```
Pearls_AQI/
├── .github/
│   └── workflows/
│       ├── feature-pipeline.yml
│       ├── training-pipeline.yml
│       └── tests.yml
├── api/
│   ├── fetchers/
│   │   ├── base.py
│   │   └── providers.py
│   ├── orchestrator.py
│   └── schemas/
│       └── data_models.py
├── config/
│   └── settings.py
├── dashboard/
│   └── app.py
├── data/
│   └── processed/
├── features/
│   ├── feature_engineering.py
│   └── main.py
├── models/
│   ├── trainer.py
│   ├── training_pipeline.py
│   └── karachi/
│       ├── Lasso.pkl
│       ├── scaler.pkl
│       └── training_results.json
├── scripts/
│   ├── upload_features_to_hopsworks.py
│   ├── upload_model_to_hopsworks.py
│   └── download_features_from_hopsworks.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run_pipeline.py
```

---

## 10. Configuration

### 10.1 Environment Variables

Create a `.env` file:

```env
# Hopsworks Configuration
HOPSWORKS_API_KEY=your_hopsworks_api_key

# API Keys (Optional - some APIs have free tiers)
OPENWEATHERMAP_API_KEY=your_openweathermap_key
WEATHERAPI_KEY=your_weatherapi_key
AQICN_TOKEN=your_aqicn_token
```

### 10.2 Settings

Configuration in `config/settings.py`:
- Karachi coordinates: 24.8607° N, 67.0011° E
- AQI prediction horizon: 72 hours (3 days)
- Feature update frequency: Hourly
- Model retraining frequency: Daily

---

## 11. Results & Metrics

### 11.1 Model Performance Summary

| Metric | Value |
|--------|-------|
| Best Model | Lasso Regression |
| RMSE | 0.1008 |
| MAE | 0.0823 |
| R² Score | 0.9892 |

### 11.2 Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | 48+ |
| Date Range | Feb 2024 - Feb 2026 |
| Features | 148 |
| Missing Data | < 2% |

---

## 12. Karachi-Specific Considerations

### 12.1 Geographic Factors
- **Coastal Location**: Sea breeze effects on air quality
- **Industrial Areas**: Korangi, SITE, FB Industrial areas
- **Urban Heat Island**: Dense urban construction

### 12.2 Seasonal Patterns
- **Winter (Nov-Feb)**: High AQI due to temperature inversion
- **Monsoon (Jul-Sep)**: Lower AQI due to rainfall
- **Summer (May-Jun)**: ModerateQI with dust storms A

### 12.3 Pollution Sources
- Vehicle emissions (major contributor)
- Industrial discharge
- Construction dust
- Crop burning (seasonal)

---

## 13. Future Improvements

### 13.1 Planned Enhancements
- [ ] Add more data sources (satellite data)
- [ ] Implement deep learning models (LSTM, Transformer)
- [ ] Add real-time notification system
- [ ] Implement model drift detection
- [ ] Add more cities

### 13.2 Potential Features
- Mobile application
- WhatsApp/Telegram bot
- API endpoint for predictions
- Custom alert thresholds

---

## 14. Troubleshooting

### 14.1 Common Issues

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Run `pip install -r requirements.txt` |
| API Key Error | Check .env file configuration |
| Data Loading Error | Verify Hopsworks connection |
| Model Not Found | Run training pipeline first |

### 14.2 Testing Connections

```bash
python scripts/test_connections.py
```

---

## 15. License & Acknowledgments

### 15.1 License
MIT License

### 15.2 Data Sources Acknowledgment
- Open-Meteo (https://open-meteo.com/)
- AQICN (https://aqicn.org/)
- OpenWeatherMap (https://openweathermap.org/)
- WeatherAPI (https://www.weatherapi.com/)

### 15.3 Tools & Platforms
- Hopsworks Feature Store
- Streamlit
- GitHub Actions
- Streamlit Community Cloud

---

## 16. Contact & Support

For questions or issues:
- GitHub Issues: https://github.com/TayyabHussain09/Pearls_AQI/issues
- Email: [Contact through GitHub]

---

**Document Version**: 1.0  
**Last Updated**: February 2026  
**Author**: Tayyab Hussain
