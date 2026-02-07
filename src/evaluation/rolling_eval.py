import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
import sqlite3

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionStore:
    """
    Stores predictions and realized labels in SQLite database
    """
    
    def __init__(self, db_path: str = "data/predictions.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE NOT NULL,
                prediction_time TIMESTAMP NOT NULL,
                predicted_proba REAL NOT NULL,
                predicted_class INTEGER NOT NULL,
                model_version TEXT NOT NULL,
                features TEXT,
                UNIQUE(prediction_date)
            )
        """)
        
        # Labels table (realized outcomes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS realized_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_date DATE NOT NULL,
                label_received_time TIMESTAMP NOT NULL,
                actual_label INTEGER NOT NULL,
                next_day_return REAL,
                UNIQUE(prediction_date)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Prediction database initialized: {self.db_path}")
    
    def store_prediction(
        self,
        prediction_date: str,
        predicted_proba: float,
        model_version: str,
        features: Optional[Dict] = None
    ):
        """
        Store a prediction for a given date
        
        Args:
            prediction_date: Date of prediction (YYYY-MM-DD)
            predicted_proba: Predicted probability of positive class
            model_version: Model version that made the prediction
            features: Optional feature values used
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        predicted_class = 1 if predicted_proba >= 0.5 else 0
        features_json = json.dumps(features) if features else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO predictions 
            (prediction_date, prediction_time, predicted_proba, predicted_class, model_version, features)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            prediction_date,
            datetime.now().isoformat(),
            predicted_proba,
            predicted_class,
            model_version,
            features_json
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored prediction for {prediction_date}: {predicted_proba:.4f}")
    
    def store_realized_label(
        self,
        prediction_date: str,
        actual_label: int,
        next_day_return: Optional[float] = None
    ):
        """
        Store the realized label for a prediction
        
        Args:
            prediction_date: Date the prediction was made for
            actual_label: Actual binary outcome (0 or 1)
            next_day_return: Optional actual return value
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO realized_labels 
            (prediction_date, label_received_time, actual_label, next_day_return)
            VALUES (?, ?, ?, ?)
        """, (
            prediction_date,
            datetime.now().isoformat(),
            actual_label,
            next_day_return
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored realized label for {prediction_date}: {actual_label}")
    
    def get_predictions_with_labels(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get predictions joined with realized labels
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with predictions and labels
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                p.prediction_date,
                p.predicted_proba,
                p.predicted_class,
                p.model_version,
                r.actual_label,
                r.next_day_return
            FROM predictions p
            INNER JOIN realized_labels r
                ON p.prediction_date = r.prediction_date
            WHERE 1=1
        """
        
        params = []
        if start_date:
            query += " AND p.prediction_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND p.prediction_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY p.prediction_date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df


class RollingEvaluator:
    """
    Computes rolling metrics for online model evaluation
    """
    
    def __init__(self, prediction_store: PredictionStore):
        self.store = prediction_store
        self.metrics_db_path = "data/rolling_metrics.db"
        self._init_metrics_db()
    
    def _init_metrics_db(self):
        """Initialize rolling metrics database"""
        Path(self.metrics_db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rolling_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_date DATE NOT NULL,
                window_days INTEGER NOT NULL,
                n_samples INTEGER NOT NULL,
                auc REAL,
                log_loss REAL,
                brier_score REAL,
                accuracy REAL,
                computed_at TIMESTAMP NOT NULL,
                UNIQUE(evaluation_date, window_days)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Rolling metrics database initialized: {self.metrics_db_path}")
    
    def compute_rolling_metrics(
        self,
        evaluation_date: str,
        window_days: int = 20
    ) -> Optional[Dict[str, float]]:
        """
        Compute metrics over a rolling window
        
        Args:
            evaluation_date: Date to compute metrics as of
            window_days: Number of days to look back
            
        Returns:
            Dictionary of metrics or None if insufficient data
        """
        # Get predictions with labels in window
        end_date = evaluation_date
        start_date = (pd.to_datetime(evaluation_date) - timedelta(days=window_days)).strftime('%Y-%m-%d')
        
        df = self.store.get_predictions_with_labels(start_date, end_date)
        
        if len(df) < 5:  # Need minimum samples
            logger.warning(f"Insufficient data for {evaluation_date} ({len(df)} samples)")
            return None
        
        # Compute metrics
        y_true = df['actual_label'].values
        y_pred_proba = df['predicted_proba'].values
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, (y_pred_proba >= 0.5).astype(int))
        }
        
        logger.info(f"Rolling metrics ({window_days}d) for {evaluation_date}:")
        logger.info(f"  Samples: {len(df)}")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")
        logger.info(f"  Brier: {metrics['brier_score']:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def store_rolling_metrics(
        self,
        evaluation_date: str,
        window_days: int,
        metrics: Dict[str, float],
        n_samples: int
    ):
        """Store computed rolling metrics"""
        conn = sqlite3.connect(self.metrics_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO rolling_metrics
            (evaluation_date, window_days, n_samples, auc, log_loss, brier_score, accuracy, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_date,
            window_days,
            n_samples,
            metrics['auc'],
            metrics['log_loss'],
            metrics['brier_score'],
            metrics['accuracy'],
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def get_rolling_metrics_history(
        self,
        window_days: int = 20,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get historical rolling metrics
        
        Args:
            window_days: Window size to filter
            limit: Maximum number of records
            
        Returns:
            DataFrame of rolling metrics over time
        """
        conn = sqlite3.connect(self.metrics_db_path)
        
        query = """
            SELECT *
            FROM rolling_metrics
            WHERE window_days = ?
            ORDER BY evaluation_date DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(window_days, limit))
        conn.close()
        
        return df.sort_values('evaluation_date')
    
    def daily_evaluation_job(self, evaluation_date: Optional[str] = None):
        """
        Daily job to compute rolling metrics
        
        Args:
            evaluation_date: Date to evaluate (defaults to today)
        """
        if evaluation_date is None:
            evaluation_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info("=" * 70)
        logger.info(f"Daily Evaluation Job - {evaluation_date}")
        logger.info("=" * 70)
        
        # Compute metrics for different windows
        windows = [20, 60]
        
        for window_days in windows:
            logger.info(f"\nComputing {window_days}-day rolling metrics...")
            
            metrics = self.compute_rolling_metrics(evaluation_date, window_days)
            
            if metrics:
                # Get sample count
                end_date = evaluation_date
                start_date = (pd.to_datetime(evaluation_date) - timedelta(days=window_days)).strftime('%Y-%m-%d')
                df = self.store.get_predictions_with_labels(start_date, end_date)
                
                self.store_rolling_metrics(
                    evaluation_date=evaluation_date,
                    window_days=window_days,
                    metrics=metrics,
                    n_samples=len(df)
                )
        
        logger.info("=" * 70)
        logger.info("Daily evaluation complete")
        logger.info("=" * 70)
    
    def plot_rolling_performance(
        self,
        window_days: int = 20,
        save_path: str = "experiments/rolling_performance.png"
    ):
        """
        Plot rolling metrics over time
        
        Args:
            window_days: Window size to plot
            save_path: Where to save the plot
        """
        df = self.get_rolling_metrics_history(window_days=window_days, limit=200)
        
        if len(df) == 0:
            logger.warning("No rolling metrics data to plot")
            return
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Rolling Performance ({window_days}-Day Window)', fontsize=16)
        
        df['evaluation_date'] = pd.to_datetime(df['evaluation_date'])
        
        metrics = [
            ('auc', 'AUC'),
            ('log_loss', 'Log Loss'),
            ('brier_score', 'Brier Score'),
            ('accuracy', 'Accuracy')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            ax.plot(df['evaluation_date'], df[metric], marker='o', linewidth=2, markersize=4)
            ax.axhline(df[metric].mean(), color='r', linestyle='--', 
                      label=f'Mean: {df[metric].mean():.4f}')
            
            ax.set_xlabel('Date')
            ax.set_ylabel(title)
            ax.set_title(f'{title} Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Rolling performance plot saved to {save_path}")
        plt.close()


def simulate_daily_predictions():
    """
    Simulate daily prediction workflow with historical data
    """
    logger.info("=" * 70)
    logger.info("SIMULATING DAILY PREDICTION WORKFLOW")
    logger.info("=" * 70)
    
    # Load historical data
    df = pd.read_csv("data/nvda_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Load feature names
    with open("data/feature_names.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    # Initialize stores
    pred_store = PredictionStore()
    evaluator = RollingEvaluator(pred_store)
    
    # Simulate predictions for last 60 days
    recent_data = df.tail(60).copy()
    
    logger.info(f"\nSimulating predictions for {len(recent_data)} days...")
    
    for idx, row in recent_data.iterrows():
        prediction_date = row['date'].strftime('%Y-%m-%d')
        
        # Simulate prediction (using a simple heuristic for demo)
        # In production, this would be your actual model prediction
        predicted_proba = 0.5 + (row['return_lag_1'] * 2)  # Simple momentum
        predicted_proba = np.clip(predicted_proba, 0.01, 0.99)
        
        # Store prediction
        pred_store.store_prediction(
            prediction_date=prediction_date,
            predicted_proba=predicted_proba,
            model_version="v0"
        )
        
        # Store realized label (we know this because we're simulating)
        pred_store.store_realized_label(
            prediction_date=prediction_date,
            actual_label=int(row['target']),
            next_day_return=row['next_day_return']
        )
    
    logger.info("✓ Predictions and labels stored")
    
    # Run daily evaluations for last 30 days
    logger.info("\nRunning daily evaluations...")
    
    eval_dates = recent_data.tail(30)['date']
    
    for eval_date in eval_dates:
        eval_date_str = eval_date.strftime('%Y-%m-%d')
        evaluator.daily_evaluation_job(evaluation_date=eval_date_str)
    
    # Plot rolling performance
    evaluator.plot_rolling_performance(window_days=20)
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ SIMULATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Databases created:")
    logger.info(f"  - Predictions: data/predictions.db")
    logger.info(f"  - Rolling metrics: data/rolling_metrics.db")
    logger.info("Plot created:")
    logger.info(f"  - experiments/rolling_performance.png")
    logger.info("=" * 70)


def main():
    """Run simulation of daily prediction and evaluation loop"""
    simulate_daily_predictions()


if __name__ == "__main__":
    main()