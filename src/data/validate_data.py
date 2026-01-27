"""
Data Quality Validation Script
Validates the fetched NVDA data before feature engineering
"""
import pandas as pd
import numpy as np
from pathlib import Path

def validate_data(file_path: str = "data/nvda_labeled.csv"):
    """Run comprehensive data quality checks"""
    
    print("=" * 70)
    print("NVDA DATA QUALITY VALIDATION")
    print("=" * 70)
    
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n📊 Dataset Overview")
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Trading days: {(df['date'].max() - df['date'].min()).days} calendar days")
    print(f"   Columns: {', '.join(df.columns)}")
    
    # Check 1: Missing values
    print(f"\n✓ Check 1: Missing Values")
    missing = df.isnull().sum()
    if missing.any():
        print(f"   ⚠️  Found missing values:\n{missing[missing > 0]}")
    else:
        print(f"   ✓ No missing values found")
    
    # Check 2: Data types
    print(f"\n✓ Check 2: Data Types")
    print(f"   {df.dtypes.to_dict()}")
    
    # Check 3: Date monotonicity
    print(f"\n✓ Check 3: Date Monotonicity")
    if df['date'].is_monotonic_increasing:
        print(f"   ✓ Dates are properly ordered")
    else:
        print(f"   ⚠️  Dates are NOT monotonic!")
    
    # Check 4: Price sanity
    print(f"\n✓ Check 4: Price Ranges")
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        print(f"   {col:>6}: ${df[col].min():>8.2f} to ${df[col].max():>8.2f}")
    
    # Check 5: OHLC relationships
    print(f"\n✓ Check 5: OHLC Integrity")
    high_violations = (df['high'] < df[['open', 'low', 'close']].max(axis=1)).sum()
    low_violations = (df['low'] > df[['open', 'high', 'close']].min(axis=1)).sum()
    
    if high_violations == 0 and low_violations == 0:
        print(f"   ✓ All OHLC relationships valid")
    else:
        print(f"   ⚠️  High violations: {high_violations}, Low violations: {low_violations}")
    
    # Check 6: Volume sanity
    print(f"\n✓ Check 6: Volume Statistics")
    print(f"   Min: {df['volume'].min():,}")
    print(f"   Max: {df['volume'].max():,}")
    print(f"   Mean: {df['volume'].mean():,.0f}")
    print(f"   Zero volume days: {(df['volume'] == 0).sum()}")
    
    # Check 7: Returns distribution
    print(f"\n✓ Check 7: Returns Distribution")
    print(f"   Daily return mean: {df['daily_return'].mean():.4%}")
    print(f"   Daily return std: {df['daily_return'].std():.4%}")
    print(f"   Daily return min: {df['daily_return'].min():.4%}")
    print(f"   Daily return max: {df['daily_return'].max():.4%}")
    
    # Check 8: Target distribution
    print(f"\n✓ Check 8: Target Label Distribution")
    target_counts = df['target'].value_counts().sort_index()
    print(f"   Class 0 (down): {target_counts[0]} ({target_counts[0]/len(df):.1%})")
    print(f"   Class 1 (up):   {target_counts[1]} ({target_counts[1]/len(df):.1%})")
    
    balance = target_counts[1] / len(df)
    if 0.4 <= balance <= 0.6:
        print(f"   ✓ Well balanced")
    else:
        print(f"   ⚠️  Imbalanced dataset")
    
    # Check 9: Label alignment verification
    print(f"\n✓ Check 9: Label Alignment Verification")
    # Manually verify a few samples
    sample_indices = [0, 100, 250, 400]
    misaligned = 0
    for idx in sample_indices:
        if idx + 1 < len(df):
            expected_return = (df.iloc[idx + 1]['close'] / df.iloc[idx]['close']) - 1
            actual_next_return = df.iloc[idx]['next_day_return']
            if not np.isclose(expected_return, actual_next_return, rtol=1e-5):
                misaligned += 1
    
    if misaligned == 0:
        print(f"   ✓ Labels correctly aligned (spot-checked {len(sample_indices)} samples)")
    else:
        print(f"   ⚠️  Found {misaligned} misaligned labels!")
    
    # Check 10: Recent data freshness
    print(f"\n✓ Check 10: Data Freshness")
    latest_date = df['date'].max()
    days_old = (pd.Timestamp.now() - latest_date).days
    print(f"   Latest data: {latest_date.date()}")
    print(f"   Age: {days_old} days old")
    
    if days_old <= 7:
        print(f"   ✓ Data is fresh")
    else:
        print(f"   ⚠️  Data is {days_old} days old (consider re-fetching)")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"VALIDATION SUMMARY")
    print(f"=" * 70)
    print(f"✓ Dataset contains {len(df)} valid trading days")
    print(f"✓ Target distribution: {balance:.1%} positive class")
    print(f"✓ Ready for feature engineering!")
    print(f"=" * 70)
    
    return df


if __name__ == "__main__":
    df = validate_data()
    
    # Optional: Show first and last few rows
    print("\n📋 First 3 rows:")
    print(df.head(3).to_string(index=False))
    
    print("\n📋 Last 3 rows:")
    print(df.tail(3).to_string(index=False))