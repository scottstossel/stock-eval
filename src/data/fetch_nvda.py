"""
Daily NVDA Stock Data Fetcher
Runs after market close (4:00 PM ET) to fetch historical OHLCV data
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NVDADataFetcher:
    """Fetches and processes NVDA stock data with data quality checks"""
    
    def __init__(self, data_dir: str = "data"):
        self.ticker = "NVDA"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.raw_file = self.data_dir / "nvda_raw.csv"
        self.labeled_file = self.data_dir / "nvda_labeled.csv"
    
    def fetch_historical_data(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for NVDA
        
        Args:
            start_date: Start date (YYYY-MM-DD). Defaults to 2 years ago
            end_date: End date (YYYY-MM-DD). Defaults to today
        
        Returns:
            DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {self.ticker} data from {start_date} to {end_date}")
        
        try:
            # Fetch data from yfinance
            df = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if df.empty:
                raise ValueError("No data returned from yfinance")
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Handle MultiIndex columns (newer yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize column names
            df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
            
            # Ensure we have the expected columns
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Got: {df.columns.tolist()}")
            
            # Keep only required columns and adj_close if available
            keep_cols = required_cols.copy()
            if 'adj_close' in df.columns:
                keep_cols.append('adj_close')
            df = df[keep_cols]
            
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def normalize_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamps and validate data quality
        
        Args:
            df: Raw dataframe
            
        Returns:
            Validated and normalized dataframe
        """
        df = df.copy()
        
        # Convert date to datetime and normalize to date only
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Data quality checks
        logger.info("Running data quality checks...")
        
        # Check for NaNs
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            logger.warning(f"Found NaN values:\n{nan_counts[nan_counts > 0]}")
            # Forward fill missing values (common for holidays)
            df = df.fillna(method='ffill')
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['date'], keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate dates")
            df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Verify monotonic dates
        if not df['date'].is_monotonic_increasing:
            logger.error("Dates are not monotonically increasing!")
            df = df.sort_values('date').reset_index(drop=True)
        
        # Check for negative prices or volumes
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if (df[col] < 0).any():
                logger.error(f"Found negative values in {col}")
        
        logger.info("Data quality checks passed")
        return df
    
    def compute_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute binary target labels and daily returns
        
        Args:
            df: Normalized dataframe
            
        Returns:
            Dataframe with returns and labels
        """
        df = df.copy()
        
        # Compute daily returns
        df['daily_return'] = df['close'].pct_change()
        
        # Compute next-day return (this is what we're predicting)
        df['next_day_return'] = df['daily_return'].shift(-1)
        
        # Binary target: 1 if next-day return > 0, else 0
        df['target'] = (df['next_day_return'] > 0).astype(int)
        
        # Drop the last row (no next-day return available)
        df = df[:-1].copy()
        
        # Verify label alignment
        logger.info("Verifying label alignment...")
        assert df['target'].notna().all(), "Found NaN in target labels"
        assert len(df) > 0, "No valid samples after labeling"
        
        label_counts = df['target'].value_counts()
        logger.info(f"Label distribution:\n{label_counts}")
        logger.info(f"Class balance: {label_counts[1] / len(df):.2%} positive")
        
        return df
    
    def save_data(self, df_raw: pd.DataFrame, df_labeled: pd.DataFrame):
        """Save raw and labeled datasets"""
        df_raw.to_csv(self.raw_file, index=False)
        logger.info(f"Saved raw data to {self.raw_file}")
        
        df_labeled.to_csv(self.labeled_file, index=False)
        logger.info(f"Saved labeled data to {self.labeled_file}")
    
    def run_daily_fetch(self):
        """Main entry point for daily data fetch"""
        logger.info("=" * 60)
        logger.info("Starting daily NVDA data fetch")
        logger.info("=" * 60)
        
        try:
            # Fetch data
            df_raw = self.fetch_historical_data()
            
            # Normalize and validate
            df_normalized = self.normalize_and_validate(df_raw)
            
            # Compute labels
            df_labeled = self.compute_labels(df_normalized)
            
            # Save both versions
            self.save_data(df_normalized, df_labeled)
            
            logger.info("=" * 60)
            logger.info("Daily fetch completed successfully")
            logger.info(f"Total records: {len(df_labeled)}")
            logger.info(f"Date range: {df_labeled['date'].min()} to {df_labeled['date'].max()}")
            logger.info("=" * 60)
            
            return df_labeled
            
        except Exception as e:
            logger.error(f"Daily fetch failed: {e}")
            raise


def main():
    """Run the daily fetch"""
    fetcher = NVDADataFetcher(data_dir="data")
    fetcher.run_daily_fetch()


if __name__ == "__main__":
    main()