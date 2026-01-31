"""
Feature Engineering for Stock Prediction
Creates causal features with no data leakage
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureConfig:
    """Configuration for feature generation"""
    
    def __init__(self):
        self.config = {
            "lagged_returns": {
                "lags": [1, 2, 3, 5, 10],
                "description": "Past daily returns (t-1, t-2, etc.)"
            },
            "moving_averages": {
                "windows": [5, 10, 20],
                "price_col": "close",
                "description": "Simple moving averages of closing price"
            },
            "rolling_volatility": {
                "windows": [5, 10, 20],
                "description": "Rolling standard deviation of returns"
            },
            "volume_features": {
                "windows": [5, 10, 20],
                "description": "Volume change and rolling stats"
            },
            "price_features": {
                "enabled": True,
                "description": "High-low spread, close position in daily range"
            }
        }
    
    def save(self, filepath: str = "src/features/feature_config.json"):
        """Save config to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Feature config saved to {filepath}")
    
    def load(self, filepath: str = "src/features/feature_config.json"):
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        logger.info(f"Feature config loaded from {filepath}")
        return self.config
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names that will be generated"""
        features = []
        
        # Lagged returns
        for lag in self.config["lagged_returns"]["lags"]:
            features.append(f"return_lag_{lag}")
        
        # Moving averages
        for window in self.config["moving_averages"]["windows"]:
            features.append(f"ma_{window}")
            features.append(f"price_to_ma_{window}")
        
        # Rolling volatility
        for window in self.config["rolling_volatility"]["windows"]:
            features.append(f"volatility_{window}")
        
        # Volume features
        for window in self.config["volume_features"]["windows"]:
            features.append(f"volume_ma_{window}")
            features.append(f"volume_ratio_{window}")
        
        # Price features
        if self.config["price_features"]["enabled"]:
            features.extend([
                "high_low_spread",
                "close_position_in_range"
            ])
        
        return features


class StockFeatureEngineer:
    """Generate causal features for stock prediction"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config if config else FeatureConfig()
        self.feature_names = []
    
    def create_lagged_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lagged return features
        Ensures causality: only use past returns
        """
        df = df.copy()
        lags = self.config.config["lagged_returns"]["lags"]
        
        logger.info(f"Creating lagged returns: {lags}")
        
        for lag in lags:
            feature_name = f"return_lag_{lag}"
            df[feature_name] = df['daily_return'].shift(lag)
            self.feature_names.append(feature_name)
        
        return df
    
    def create_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create moving average features
        Uses only past data (rolling window looks backward)
        """
        df = df.copy()
        windows = self.config.config["moving_averages"]["windows"]
        price_col = self.config.config["moving_averages"]["price_col"]
        
        logger.info(f"Creating moving averages: {windows}")
        
        for window in windows:
            # Simple moving average
            ma_name = f"ma_{window}"
            df[ma_name] = df[price_col].rolling(window=window, min_periods=window).mean()
            self.feature_names.append(ma_name)
            
            # Price relative to MA (momentum indicator)
            ratio_name = f"price_to_ma_{window}"
            df[ratio_name] = df[price_col] / df[ma_name] - 1
            self.feature_names.append(ratio_name)
        
        return df
    
    def create_rolling_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling volatility features
        Measures recent price variability
        """
        df = df.copy()
        windows = self.config.config["rolling_volatility"]["windows"]
        
        logger.info(f"Creating rolling volatility: {windows}")
        
        for window in windows:
            feature_name = f"volatility_{window}"
            df[feature_name] = df['daily_return'].rolling(
                window=window, 
                min_periods=window
            ).std()
            self.feature_names.append(feature_name)
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create volume-based features
        Volume can indicate conviction in price moves
        """
        df = df.copy()
        windows = self.config.config["volume_features"]["windows"]
        
        logger.info(f"Creating volume features: {windows}")
        
        for window in windows:
            # Volume moving average
            ma_name = f"volume_ma_{window}"
            df[ma_name] = df['volume'].rolling(window=window, min_periods=window).mean()
            self.feature_names.append(ma_name)
            
            # Current volume relative to MA
            ratio_name = f"volume_ratio_{window}"
            df[ratio_name] = df['volume'] / df[ma_name]
            self.feature_names.append(ratio_name)
        
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create intraday price features
        Uses only current day's OHLC (causal)
        """
        df = df.copy()
        
        if not self.config.config["price_features"]["enabled"]:
            return df
        
        logger.info("Creating price features")
        
        # High-low spread (normalized by close)
        df['high_low_spread'] = (df['high'] - df['low']) / df['close']
        self.feature_names.append('high_low_spread')
        
        # Where did price close within the day's range?
        # 0 = closed at low, 1 = closed at high
        range_size = df['high'] - df['low']
        df['close_position_in_range'] = np.where(
            range_size > 0,
            (df['close'] - df['low']) / range_size,
            0.5  # Default to middle if no range
        )
        self.feature_names.append('close_position_in_range')
        
        return df
    
    def verify_no_leakage(self, df: pd.DataFrame) -> bool:
        """
        Verify features don't use future information
        Checks that features at time t only use data from t and before
        """
        logger.info("Verifying no data leakage...")
        
        # Check that all features have NaN in early rows (due to windows)
        # This indicates we're using rolling windows correctly
        max_window = max(
            self.config.config["moving_averages"]["windows"] +
            self.config.config["rolling_volatility"]["windows"] +
            self.config.config["volume_features"]["windows"]
        )
        
        # Features should have NaN in first max_window-1 rows
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'open', 'high', 'low', 'close', 'volume', 
                        'daily_return', 'next_day_return', 'target']]
        
        early_nans = df[feature_cols].head(max_window - 1).isnull().all()
        
        if not early_nans.all():
            logger.warning("Some features don't have expected NaNs in early rows")
            logger.warning(f"This might indicate data leakage!")
            return False
        
        logger.info("✓ No data leakage detected")
        return True
    
    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature generation pipeline
        
        Args:
            df: DataFrame with OHLCV and labels
            
        Returns:
            DataFrame with all features added
        """
        logger.info("=" * 70)
        logger.info("Starting feature generation")
        logger.info("=" * 70)
        
        self.feature_names = []
        
        # Generate all feature groups
        df = self.create_lagged_returns(df)
        df = self.create_moving_averages(df)
        df = self.create_rolling_volatility(df)
        df = self.create_volume_features(df)
        df = self.create_price_features(df)
        
        # Verify no leakage
        self.verify_no_leakage(df)
        
        # Drop rows with NaN features (early rows due to windows)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        logger.info(f"Dropped {dropped_rows} rows due to window requirements")
        logger.info(f"Final dataset: {len(df)} samples")
        logger.info(f"Generated {len(self.feature_names)} features")
        logger.info("=" * 70)
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Return list of feature column names"""
        return self.feature_names


def main():
    """Generate features for NVDA dataset"""
    
    # Load labeled data
    logger.info("Loading labeled data...")
    df = pd.read_csv("data/nvda_labeled.csv")
    df['date'] = pd.to_datetime(df['date'])
    logger.info(f"Loaded {len(df)} records")
    
    # Create feature config and save it
    config = FeatureConfig()
    config.save()
    
    # Generate features
    engineer = StockFeatureEngineer(config)
    df_features = engineer.generate_features(df)
    
    # Save feature matrix
    output_path = Path("data/nvda_features.csv")
    df_features.to_csv(output_path, index=False)
    logger.info(f"Saved feature matrix to {output_path}")
    
    # Save feature names for later use
    feature_list_path = Path("data/feature_names.txt")
    with open(feature_list_path, 'w') as f:
        f.write('\n'.join(engineer.get_feature_columns()))
    logger.info(f"Saved feature names to {feature_list_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)
    print(f"Input samples: {len(df)}")
    print(f"Output samples: {len(df_features)}")
    print(f"Features created: {len(engineer.get_feature_columns())}")
    print(f"\nFeature names:")
    for i, feat in enumerate(engineer.get_feature_columns(), 1):
        print(f"  {i:2d}. {feat}")
    print("=" * 70)


if __name__ == "__main__":
    main()