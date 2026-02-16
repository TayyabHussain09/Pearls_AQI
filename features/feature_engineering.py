"""
Feature Engineering module for AQI Prediction System.
Implements advanced feature transformations including:
- Cyclical time encodings (Sin/Cos)
- Rolling averages (24h, 7d)
- Lag features for temporal dependencies
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class CyclicalTimeEncoder:
    """
    Encodes time features using cyclical sine and cosine transformations.
    Preserves the circular nature of time features.
    """
    
    def __init__(self, columns: List[str] = None):
        """
        Initialize the cyclical encoder.
        
        Args:
            columns: List of column names to encode
        """
        self.columns = columns or [
            "hour", "day_of_week", "month", "day_of_year", "week_of_year"
        ]
        self._fitted = False
    
    def _encode(self, values: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply cyclical encoding to values.
        
        Args:
            values: Array of values to encode
            period: Period of the cycle (e.g., 24 for hours)
            
        Returns:
            Tuple of (sin_encoded, cos_encoded) arrays
        """
        sin_encoded = np.sin(2 * np.pi * values / period)
        cos_encoded = np.cos(2 * np.pi * values / period)
        return sin_encoded, cos_encoded
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe with cyclical encodings.
        
        Args:
            df: Input dataframe with datetime column
            
        Returns:
            Dataframe with cyclical encoded features
        """
        df = df.copy()
        
        for col in self.columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping")
                continue
            
            if col == "hour":
                df[f"{col}_sin"], df[f"{col}_cos"] = self._encode(df[col].values, 24)
            elif col == "day_of_week":
                df[f"{col}_sin"], df[f"{col}_cos"] = self._encode(df[col].values, 7)
            elif col == "day_of_month":
                df[f"{col}_sin"], df[f"{col}_cos"] = self._encode(df[col].values, 31)
            elif col == "month":
                df[f"{col}_sin"], df[f"{col}_cos"] = self._encode(df[col].values, 12)
            elif col == "day_of_year":
                df[f"{col}_sin"], df[f"{col}_cos"] = self._encode(df[col].values, 365)
            elif col == "week_of_year":
                df[f"{col}_sin"], df[f"{col}_cos"] = self._encode(df[col].values, 52)
        
        self._fitted = True
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        return self.transform(df)


class RollingFeatureGenerator:
    """
    Generates rolling window features for time series data.
    Includes 24-hour and 7-day rolling averages.
    """
    
    def __init__(
        self,
        windows_24h: List[int] = None,
        windows_7d: List[int] = None,
        min_periods: int = 1
    ):
        """
        Initialize the rolling feature generator.
        
        Args:
            windows_24h: Windows for 24-hour features (in hours)
            windows_7d: Windows for 7-day features (in hours)
            min_periods: Minimum periods for valid window
        """
        self.windows_24h = windows_24h or [1, 3, 6, 12, 24]
        self.windows_7d = windows_7d or [48, 72, 120, 168]  # 2d, 3d, 5d, 7d
        self.min_periods = min_periods
        self._fitted = False
    
    def _create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: List[int],
        prefix: str
    ) -> pd.DataFrame:
        """
        Create rolling features for a target column.
        
        Args:
            df: Input dataframe
            target_col: Column to create features from
            windows: List of window sizes
            prefix: Prefix for new column names
            
        Returns:
            Dataframe with rolling features
        """
        df = df.copy()
        
        for window in windows:
            # Mean
            df[f"{prefix}_rolling_mean_{window}h"] = df[target_col].rolling(
                window=window, min_periods=self.min_periods
            ).mean()
            
            # Std
            df[f"{prefix}_rolling_std_{window}h"] = df[target_col].rolling(
                window=window, min_periods=self.min_periods
            ).std()
            
            # Min/Max
            df[f"{prefix}_rolling_min_{window}h"] = df[target_col].rolling(
                window=window, min_periods=self.min_periods
            ).min()
            
            df[f"{prefix}_rolling_max_{window}h"] = df[target_col].rolling(
                window=window, min_periods=self.min_periods
            ).max()
            
            # Range
            df[f"{prefix}_rolling_range_{window}h"] = (
                df[f"{prefix}_rolling_max_{window}h"] - 
                df[f"{prefix}_rolling_min_{window}h"]
            )
        
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe with rolling features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with rolling features
        """
        df = df.copy()
        target_col = settings.TARGET_COLUMN
        
        # 24-hour rolling features
        df = self._create_rolling_features(
            df, target_col, self.windows_24h, "aqi"
        )
        
        # 7-day rolling features
        df = self._create_rolling_features(
            df, target_col, self.windows_7d, "aqi"
        )
        
        self._fitted = True
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        return self.transform(df)


class LagFeatureGenerator:
    """
    Generates lag features to capture temporal dependencies.
    Creates lagged versions of key features.
    """
    
    def __init__(
        self,
        lags: List[int] = None,
        target_col: str = None
    ):
        """
        Initialize the lag feature generator.
        
        Args:
            lags: List of lag periods in hours
            target_col: Target column for lag features
        """
        self.lags = lags or [1, 2, 3, 6, 12, 24, 48, 72, 168]  # 1h, 2h, 3h, 6h, 12h, 24h, 48h, 72h, 7d
        self.target_col = target_col or settings.TARGET_COLUMN
        self._fitted = False
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe with lag features.
        
        Args:
            df: Input dataframe with datetime index or column
            
        Returns:
            Dataframe with lag features
        """
        df = df.copy()
        
        for lag in self.lags:
            # AQI lag features
            df[f"aqi_lag_{lag}h"] = df[self.target_col].shift(lag)
            
            # Weather lag features (key pollutants)
            for col in ["pm25", "pm10", "no2", "o3"]:
                if col in df.columns:
                    df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
        
        self._fitted = True
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Transformed dataframe
        """
        return self.transform(df)


class FeatureEngineer:
    """
    Main feature engineering class that orchestrates all transformations.
    """
    
    def __init__(self):
        """Initialize all feature generators."""
        self.time_encoder = CyclicalTimeEncoder()
        self.rolling_generator = RollingFeatureGenerator()
        self.lag_generator = LagFeatureGenerator()
        self.scaler = None
        self.feature_names = None
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features from datetime column.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with time features
        """
        df = df.copy()
        
        # Ensure datetime column exists
        if settings.TIME_COLUMN not in df.columns:
            raise ValueError(f"Datetime column '{settings.TIME_COLUMN}' not found")
        
        # Parse datetime if string
        if df[settings.TIME_COLUMN].dtype == 'object':
            df[settings.TIME_COLUMN] = pd.to_datetime(df[settings.TIME_COLUMN])
        
        # Extract time features
        df['hour'] = df[settings.TIME_COLUMN].dt.hour
        df['day_of_week'] = df[settings.TIME_COLUMN].dt.dayofweek
        df['day_of_month'] = df[settings.TIME_COLUMN].dt.day
        df['month'] = df[settings.TIME_COLUMN].dt.month
        df['day_of_year'] = df[settings.TIME_COLUMN].dt.dayofyear
        df['year'] = df[settings.TIME_COLUMN].dt.year
        df['week_of_year'] = df[settings.TIME_COLUMN].dt.isocalendar().week.astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        return df
    
    def add_weather_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather interaction features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with interaction features
        """
        df = df.copy()
        
        # Temperature-humidity interaction
        if 'temperature_2m' in df.columns and 'relative_humidity_2m' in df.columns:
            df['temp_humidity_interaction'] = (
                df['temperature_2m'] * df['relative_humidity_2m']
            )
        
        # Wind-pressure interaction
        if 'wind_speed_10m' in df.columns and 'pressure_msl' in df.columns:
            df['wind_pressure_interaction'] = (
                df['wind_speed_10m'] * df['pressure_msl'] / 1000
            )
        
        # Visibility-humidity interaction
        if 'visibility' in df.columns and 'relative_humidity_2m' in df.columns:
            df['visibility_humidity_interaction'] = (
                df['visibility'] / (df['relative_humidity_2m'] + 1)
            )
        
        # Karachi-specific: Sea breeze effect (wind from sea direction = West)
        if 'wind_direction_10m' in df.columns:
            df['wind_from_sea'] = ((df['wind_direction_10m'] >= 180) & 
                                   (df['wind_direction_10m'] <= 270)).astype(int)
        
        # Temperature inversion indicator (cold night - high pressure, low clouds)
        if all(c in df.columns for c in ['pressure_msl', 'cloud_cover', 'temperature_2m']):
            df['temp_inversion_risk'] = (
                (df['pressure_msl'] > 1015).astype(int) *
                (df['cloud_cover'] < 30).astype(int) *
                (df['temperature_2m'] < 20).astype(int)
            )
        
        # Pollution accumulation potential (low wind, high pressure, low precipitation)
        if all(c in df.columns for c in ['wind_speed_10m', 'pressure_msl', 'precipitation']):
            df['pollution_accumulation_risk'] = (
                (df['wind_speed_10m'] < 5).astype(int) *
                (df['pressure_msl'] > 1010).astype(int) *
                (df['precipitation'] == 0).astype(int)
            )
        
        # Seasonal monsoon indicator (June-September)
        if 'month' in df.columns:
            df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
            df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
            df['is_summer'] = df['month'].isin([5, 6, 7]).astype(int)
        
        return df
    
    def add_aqi_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add AQI category encoding.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with AQI category features
        """
        df = df.copy()
        
        if settings.TARGET_COLUMN in df.columns:
            aqi = df[settings.TARGET_COLUMN]
            
            # AQI categories
            df['aqi_good'] = (aqi <= 50).astype(int)
            df['aqi_moderate'] = ((aqi > 50) & (aqi <= 100)).astype(int)
            df['aqi_unhealthy_sensitive'] = ((aqi > 100) & (aqi <= 150)).astype(int)
            df['aqi_unhealthy'] = ((aqi > 150) & (aqi <= 200)).astype(int)
            df['aqi_very_unhealthy'] = ((aqi > 200) & (aqi <= 300)).astype(int)
            df['aqi_hazardous'] = (aqi > 300).astype(int)
        
        return df
    
    def add_change_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add change rate (first difference) features for AQI.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with change rate features
        """
        df = df.copy()
        target_col = settings.TARGET_COLUMN
        
        if target_col in df.columns:
            # Change rate features (first difference)
            df['aqi_change_1h'] = df[target_col].diff(1)
            df['aqi_change_24h'] = df[target_col].diff(24)
            
            # Trend features (24-hour difference)
            df['aqi_diff_24h'] = df[target_col].diff(24)
        
        return df
    
    def add_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Exponential Moving Average features for AQI.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with EMA features
        """
        df = df.copy()
        target_col = settings.TARGET_COLUMN
        
        if target_col in df.columns:
            # EMA features with different spans
            df['aqi_ema_24h'] = df[target_col].ewm(span=24, adjust=False).mean()
            df['aqi_ema_168h'] = df[target_col].ewm(span=168, adjust=False).mean()  # 7 days
        
        return df
    
    def add_long_term_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add long-term rolling features (30 days and 90 days).
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with long-term rolling features
        """
        df = df.copy()
        target_col = settings.TARGET_COLUMN
        
        if target_col in df.columns:
            # 30-day rolling mean (720 hours)
            df['rolling_mean_720h'] = df[target_col].rolling(
                window=720, min_periods=1
            ).mean()
            
            # 90-day rolling mean (2160 hours)
            df['rolling_mean_2160h'] = df[target_col].rolling(
                window=2160, min_periods=1
            ).mean()
        
        return df
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Pakistan-specific holiday features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with holiday features
        """
        df = df.copy()
        
        # Ensure datetime column exists
        if settings.TIME_COLUMN not in df.columns:
            return df
        
        # Parse datetime if needed
        if df[settings.TIME_COLUMN].dtype == 'object':
            df[settings.TIME_COLUMN] = pd.to_datetime(df[settings.TIME_COLUMN])
        
        # Pakistan-specific holidays (approximate dates)
        # Eid al-Fitr: varies (around April-May)
        # Eid al-Adha: varies (around June-July)
        # Independence Day: August 14
        # National Day: March 23
        # Defence Day: September 6
        # Youm-e-Iqbal: November 9
        
        holidays = []
        
        # Define holidays for 2024 and 2025
        for year in [2024, 2025]:
            # Independence Day
            holidays.append(pd.Timestamp(f'{year}-08-14'))
            # National Day (Pakistan Day)
            holidays.append(pd.Timestamp(f'{year}-03-23'))
            # Defence Day
            holidays.append(pd.Timestamp(f'{year}-09-06'))
            # Youm-e-Iqbal
            holidays.append(pd.Timestamp(f'{year}-11-09'))
            # Eid al-Fitr 2024 (approximate)
            if year == 2024:
                holidays.append(pd.Timestamp('2024-04-10'))
                holidays.append(pd.Timestamp('2024-04-11'))
                holidays.append(pd.Timestamp('2024-04-12'))
            # Eid al-Fitr 2025
            if year == 2025:
                holidays.append(pd.Timestamp('2025-03-30'))
                holidays.append(pd.Timestamp('2025-03-31'))
                holidays.append(pd.Timestamp('2025-04-01'))
            # Eid al-Adha 2024
            if year == 2024:
                holidays.append(pd.Timestamp('2024-06-16'))
                holidays.append(pd.Timestamp('2024-06-17'))
                holidays.append(pd.Timestamp('2024-06-18'))
            # Eid al-Adha 2025
            if year == 2025:
                holidays.append(pd.Timestamp('2025-06-06'))
                holidays.append(pd.Timestamp('2025-06-07'))
                holidays.append(pd.Timestamp('2025-06-08'))
        
        # Create holiday indicator
        holiday_set = set(holidays)
        df['is_holiday'] = df[settings.TIME_COLUMN].dt.normalize().isin(holiday_set).astype(int)
        
        return df
    
    def add_pollutant_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pollutant ratio features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with pollutant ratio features
        """
        df = df.copy()
        
        # PM2.5 to PM10 ratio
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)
        
        # NO2 to O3 ratio
        if 'no2' in df.columns and 'o3' in df.columns:
            df['no2_o3_ratio'] = df['no2'] / (df['o3'] + 1e-6)
        
        return df
    
    def add_composite_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add composite pollution index as weighted combination.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with composite pollution index
        """
        df = df.copy()
        
        # Calculate weighted pollution index
        # PM2.5 is most harmful, then PM10, then NO2, then O3
        weights = {
            'pm25': 0.4,
            'pm10': 0.3,
            'no2': 0.2,
            'o3': 0.1
        }
        
        available_weights = {k: v for k, v in weights.items() if k in df.columns}
        
        if available_weights:
            # Normalize weights to sum to 1
            total_weight = sum(available_weights.values())
            normalized_weights = {k: v/total_weight for k, v in available_weights.items()}
            
            # Calculate composite index
            df['pollution_index'] = sum(
                df[col] * weight for col, weight in normalized_weights.items()
            )
        
        return df
    
    def full_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            df: Raw input dataframe
            
        Returns:
            Fully engineered dataframe
        """
        logger.info("Starting feature engineering pipeline")
        
        # Step 1: Add time features
        df = self.add_time_features(df)
        logger.info("Added time features")
        
        # Step 2: Apply cyclical encoding
        df = self.time_encoder.transform(df)
        logger.info("Applied cyclical encoding")
        
        # Step 3: Add weather interactions
        df = self.add_weather_interactions(df)
        logger.info("Added weather interactions")
        
        # Step 4: Add AQI categories
        df = self.add_aqi_category(df)
        logger.info("Added AQI categories")
        
        # Step 5: Add change rate features
        df = self.add_change_rate_features(df)
        logger.info("Added change rate features")
        
        # Step 6: Add EMA features
        df = self.add_ema_features(df)
        logger.info("Added EMA features")
        
        # Step 7: Add long-term rolling features
        df = self.add_long_term_rolling_features(df)
        logger.info("Added long-term rolling features")
        
        # Step 8: Add holiday features
        df = self.add_holiday_features(df)
        logger.info("Added holiday features")
        
        # Step 9: Add pollutant ratios
        df = self.add_pollutant_ratios(df)
        logger.info("Added pollutant ratios")
        
        # Step 10: Add composite index
        df = self.add_composite_index(df)
        logger.info("Added composite index")
        
        # Step 11: Add rolling features
        df = self.rolling_generator.transform(df)
        logger.info("Added rolling features")
        
        # Step 12: Add lag features
        df = self.lag_generator.transform(df)
        logger.info("Added lag features")
        
        # Store feature names (excluding target and datetime)
        exclude_cols = [settings.TIME_COLUMN, settings.TARGET_COLUMN]
        self.feature_names = [
            col for col in df.columns 
            if col not in exclude_cols and not col.endswith('_lag_0h')
        ]
        
        logger.info(f"Feature engineering complete. Total features: {len(self.feature_names)}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Engineered dataframe
        """
        return self.full_pipeline(df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataframe (requires fit first).
        
        Args:
            df: Input dataframe
            
        Returns:
            Engineered dataframe
        """
        return self.full_pipeline(df)
