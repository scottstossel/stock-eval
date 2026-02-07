"""
Drift & Reliability Monitoring System
Detects feature drift, prediction drift, and model degradation
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime, timedelta
import sqlite3

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitors data drift and model reliability
    """
    
    def __init__(
        self,
        reference_window_days: int = 252,  # Training window (1 year)
        production_window_days: int = 21    # Recent production window (1 month)
    ):
        self.reference_window_days = reference_window_days
        self.production_window_days = production_window_days
        self.drift_db_path = "data/drift_monitoring.db"
        self._init_drift_db()
    
    def _init_drift_db(self):
        """Initialize drift monitoring database"""
        Path(self.drift_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.drift_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date DATE NOT NULL,
                reference_start DATE NOT NULL,
                reference_end DATE NOT NULL,
                production_start DATE NOT NULL,
                production_end DATE NOT NULL,
                dataset_drift_detected INTEGER NOT NULL,
                drift_share REAL,
                n_drifted_features INTEGER,
                health_status TEXT NOT NULL,
                report_path TEXT,
                computed_at TIMESTAMP NOT NULL,
                UNIQUE(report_date)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_drift (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date DATE NOT NULL,
                feature_name TEXT NOT NULL,
                drift_detected INTEGER NOT NULL,
                drift_score REAL,
                stattest_name TEXT,
                UNIQUE(report_date, feature_name)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Drift monitoring database initialized: {self.drift_db_path}")
    
    def get_reference_data(
        self,
        df: pd.DataFrame,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get reference dataset (training window)
        
        Args:
            df: Full dataset
            end_date: End date of reference period
            
        Returns:
            Reference dataset
        """
        end = pd.to_datetime(end_date)
        start = end - timedelta(days=self.reference_window_days)
        
        mask = (df['date'] >= start) & (df['date'] <= end)
        reference = df[mask].copy()
        
        logger.info(f"Reference data: {len(reference)} samples from {start.date()} to {end.date()}")
        
        return reference
    
    def get_production_data(
        self,
        df: pd.DataFrame,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get recent production dataset
        
        Args:
            df: Full dataset
            end_date: End date of production period
            
        Returns:
            Production dataset
        """
        end = pd.to_datetime(end_date)
        start = end - timedelta(days=self.production_window_days)
        
        mask = (df['date'] > end - timedelta(days=self.reference_window_days)) & \
               (df['date'] >= start) & (df['date'] <= end)
        production = df[mask].copy()
        
        logger.info(f"Production data: {len(production)} samples from {start.date()} to {end.date()}")
        
        return production
    
    def compute_drift_report(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'target',
        prediction_col: Optional[str] = None
    ) -> Report:
        """
        Generate Evidently drift report
        
        Args:
            reference_data: Reference dataset
            production_data: Production dataset
            feature_cols: Feature column names
            target_col: Target column name
            prediction_col: Optional prediction column
            
        Returns:
            Evidently Report object
        """
        logger.info("Computing drift report...")
        
        # Prepare data - keep only features and target
        ref_data = reference_data[feature_cols + [target_col]].copy()
        prod_data = production_data[feature_cols + [target_col]].copy()
        
        # Ensure all feature columns are numeric
        for col in feature_cols:
            ref_data[col] = pd.to_numeric(ref_data[col], errors='coerce')
            prod_data[col] = pd.to_numeric(prod_data[col], errors='coerce')
        
        # Column mapping for Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = target_col
        column_mapping.numerical_features = feature_cols
        
        if prediction_col and prediction_col in prod_data.columns:
            column_mapping.prediction = prediction_col
        
        # Create report with appropriate stattest
        report = Report(metrics=[
            DataDriftPreset(),  # Let Evidently choose appropriate tests
            DatasetDriftMetric(),
            DatasetMissingValuesMetric()
        ])
        
        # Run report
        report.run(
            reference_data=ref_data,
            current_data=prod_data,
            column_mapping=column_mapping
        )
        
        logger.info("✓ Drift report computed")
        
        return report
    
    def extract_drift_metrics(self, report: Report) -> Dict:
        """
        Extract key metrics from Evidently report
        
        Args:
            report: Evidently Report object
            
        Returns:
            Dictionary of drift metrics
        """
        # Get report as dictionary
        report_dict = report.as_dict()
        
        # Extract dataset drift
        dataset_drift = None
        drift_share = None
        drifted_features = []
        
        for metric in report_dict['metrics']:
            if metric['metric'] == 'DatasetDriftMetric':
                result = metric['result']
                dataset_drift = result.get('dataset_drift', False)
                drift_share = result.get('drift_share', 0.0)
                
                # Get drifted features
                if 'drift_by_columns' in result:
                    for col, col_result in result['drift_by_columns'].items():
                        if col_result.get('drift_detected', False):
                            drifted_features.append({
                                'feature': col,
                                'drift_score': col_result.get('drift_score', 0.0),
                                'stattest': col_result.get('stattest_name', 'unknown')
                            })
        
        metrics = {
            'dataset_drift_detected': dataset_drift,
            'drift_share': drift_share,
            'n_drifted_features': len(drifted_features),
            'drifted_features': drifted_features
        }
        
        return metrics
    
    def determine_health_status(
        self,
        drift_metrics: Dict,
        performance_metrics: Optional[Dict] = None
    ) -> str:
        """
        Determine model health status based on drift and performance
        
        Args:
            drift_metrics: Drift metrics
            performance_metrics: Optional performance metrics
            
        Returns:
            Health status: 'GREEN', 'YELLOW', or 'RED'
        """
        # Check drift
        dataset_drift = drift_metrics.get('dataset_drift_detected', False)
        drift_share = drift_metrics.get('drift_share', 0.0)
        
        # RED: Significant drift detected
        if dataset_drift and drift_share > 0.5:
            return 'RED'
        
        # YELLOW: Some drift detected
        if dataset_drift or drift_share > 0.3:
            return 'YELLOW'
        
        # Check performance if available
        if performance_metrics:
            auc_mean = performance_metrics.get('auc_mean', 0.5)
            
            # RED: Performance below baseline
            if auc_mean < 0.55:
                return 'RED'
            
            # YELLOW: Performance degrading
            if auc_mean < 0.60:
                return 'YELLOW'
        
        # GREEN: All good
        return 'GREEN'
    
    def save_drift_report(
        self,
        report_date: str,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        drift_metrics: Dict,
        health_status: str,
        report: Report
    ):
        """
        Save drift report to database and HTML
        
        Args:
            report_date: Date of the report
            reference_data: Reference dataset
            production_data: Production dataset
            drift_metrics: Extracted drift metrics
            health_status: Model health status
            report: Evidently Report object
        """
        # Save HTML report
        reports_dir = Path("experiments/drift_reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = reports_dir / f"drift_report_{report_date}.html"
        report.save_html(str(report_path))
        
        logger.info(f"HTML report saved: {report_path}")
        
        # Save to database
        conn = sqlite3.connect(self.drift_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO drift_reports
            (report_date, reference_start, reference_end, production_start, production_end,
             dataset_drift_detected, drift_share, n_drifted_features, health_status, 
             report_path, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_date,
            reference_data['date'].min().strftime('%Y-%m-%d'),
            reference_data['date'].max().strftime('%Y-%m-%d'),
            production_data['date'].min().strftime('%Y-%m-%d'),
            production_data['date'].max().strftime('%Y-%m-%d'),
            int(drift_metrics['dataset_drift_detected']),
            drift_metrics['drift_share'],
            drift_metrics['n_drifted_features'],
            health_status,
            str(report_path),
            datetime.now().isoformat()
        ))
        
        # Save feature drift details
        for feature_drift in drift_metrics['drifted_features']:
            cursor.execute("""
                INSERT OR REPLACE INTO feature_drift
                (report_date, feature_name, drift_detected, drift_score, stattest_name)
                VALUES (?, ?, ?, ?, ?)
            """, (
                report_date,
                feature_drift['feature'],
                1,
                feature_drift['drift_score'],
                feature_drift['stattest']
            ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Drift metrics saved to database")
    
    def get_drift_history(self, limit: int = 30) -> pd.DataFrame:
        """
        Get historical drift reports
        
        Args:
            limit: Maximum number of reports
            
        Returns:
            DataFrame of drift history
        """
        conn = sqlite3.connect(self.drift_db_path)
        
        query = """
            SELECT *
            FROM drift_reports
            ORDER BY report_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df.sort_values('report_date')
    
    def generate_drift_summary(
        self,
        report_date: str,
        df_full: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict:
        """
        Generate complete drift analysis for a date
        
        Args:
            report_date: Date to analyze
            df_full: Full dataset
            feature_cols: Feature columns
            
        Returns:
            Drift summary dictionary
        """
        logger.info("=" * 70)
        logger.info(f"Drift Analysis - {report_date}")
        logger.info("=" * 70)
        
        # Get reference and production data
        reference_data = self.get_reference_data(df_full, report_date)
        production_data = self.get_production_data(df_full, report_date)
        
        if len(production_data) == 0:
            logger.warning("No production data available")
            return None
        
        # Compute drift report
        report = self.compute_drift_report(
            reference_data=reference_data,
            production_data=production_data,
            feature_cols=feature_cols
        )
        
        # Extract metrics
        drift_metrics = self.extract_drift_metrics(report)
        
        # Determine health status
        health_status = self.determine_health_status(drift_metrics)
        
        # Log summary
        logger.info(f"\nDrift Detection Results:")
        logger.info(f"  Dataset drift detected: {drift_metrics['dataset_drift_detected']}")
        logger.info(f"  Drift share: {drift_metrics['drift_share']:.2%}")
        logger.info(f"  Drifted features: {drift_metrics['n_drifted_features']}/{len(feature_cols)}")
        
        if drift_metrics['drifted_features']:
            logger.info(f"\n  Drifted features:")
            for feat in drift_metrics['drifted_features']:
                logger.info(f"    - {feat['feature']}: score={feat['drift_score']:.4f}")
        
        logger.info(f"\n  Health Status: {health_status}")
        
        # Save report
        self.save_drift_report(
            report_date=report_date,
            reference_data=reference_data,
            production_data=production_data,
            drift_metrics=drift_metrics,
            health_status=health_status,
            report=report
        )
        
        summary = {
            'report_date': report_date,
            'health_status': health_status,
            'drift_metrics': drift_metrics,
            'reference_samples': len(reference_data),
            'production_samples': len(production_data)
        }
        
        logger.info("=" * 70)
        
        return summary


def demo_drift_monitoring():
    """
    Demo: Run drift monitoring on historical data
    """
    logger.info("=" * 70)
    logger.info("DRIFT MONITORING DEMO")
    logger.info("=" * 70)
    
    # Load data
    df = pd.read_csv("data/nvda_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    with open("data/feature_names.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Initialize monitor
    monitor = DriftMonitor(
        reference_window_days=252,  # 1 year reference
        production_window_days=21   # 1 month production
    )
    
    # Run drift analysis for last 5 dates (every 10 days)
    analysis_dates = df['date'].tail(50).iloc[::10]
    
    summaries = []
    
    for analysis_date in analysis_dates:
        date_str = analysis_date.strftime('%Y-%m-%d')
        
        summary = monitor.generate_drift_summary(
            report_date=date_str,
            df_full=df,
            feature_cols=feature_cols
        )
        
        if summary:
            summaries.append(summary)
    
    # Print overall summary
    logger.info("\n" + "=" * 70)
    logger.info("DRIFT MONITORING SUMMARY")
    logger.info("=" * 70)
    
    for summary in summaries:
        status_emoji = {
            'GREEN': '✓',
            'YELLOW': '⚠',
            'RED': '✗'
        }[summary['health_status']]
        
        logger.info(f"\n{status_emoji} {summary['report_date']} - {summary['health_status']}")
        logger.info(f"   Drift detected: {summary['drift_metrics']['dataset_drift_detected']}")
        logger.info(f"   Drift share: {summary['drift_metrics']['drift_share']:.2%}")
        logger.info(f"   Drifted features: {summary['drift_metrics']['n_drifted_features']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Drift monitoring complete!")
    logger.info("=" * 70)
    logger.info("Outputs:")
    logger.info(f"  - Database: {monitor.drift_db_path}")
    logger.info(f"  - HTML reports: experiments/drift_reports/")
    logger.info("=" * 70)


if __name__ == "__main__":
    demo_drift_monitoring()