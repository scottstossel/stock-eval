"""
Walk-Forward Evaluation System
Implements time-series cross-validation with proper temporal ordering
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json
from datetime import datetime
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    brier_score_loss, 
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """
    Creates walk-forward splits for time-series cross-validation
    Ensures no future data leaks into training
    """
    
    def __init__(
        self, 
        train_window: int = 252,  # ~1 year of trading days
        test_window: int = 21,     # ~1 month
        step_size: int = 21        # Move forward by 1 month
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits
        
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        n = len(df)
        
        # Start position for first test set
        start = self.train_window
        
        while start + self.test_window <= n:
            # Train on window before test
            train_start = start - self.train_window
            train_end = start
            train_idx = np.arange(train_start, train_end)
            
            # Test on next window
            test_start = start
            test_end = start + self.test_window
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
            
            # Move forward
            start += self.step_size
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        logger.info(f"Train window: {self.train_window}, Test window: {self.test_window}")
        logger.info(f"Step size: {self.step_size}")
        
        return splits


class ModelEvaluator:
    """Evaluate model predictions with multiple metrics"""
    
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for class 1
            
        Returns:
            Dictionary of metric names and values
        """
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
        }
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = ""):
        """Pretty print metrics"""
        print(f"{prefix}Metrics:")
        print(f"  AUC:          {metrics['auc']:.4f}")
        print(f"  Log Loss:     {metrics['log_loss']:.4f}")
        print(f"  Brier Score:  {metrics['brier_score']:.4f}")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")


class WalkForwardValidator:
    """Main walk-forward validation orchestrator"""
    
    def __init__(
        self,
        train_window: int = 252,
        test_window: int = 21,
        step_size: int = 21
    ):
        self.splitter = WalkForwardSplitter(train_window, test_window, step_size)
        self.evaluator = ModelEvaluator()
        self.results = []
    
    def train_baseline_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> LogisticRegression:
        """
        Train baseline logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train, y_train)
        return model
    
    def run_validation(
        self, 
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'target'
    ) -> pd.DataFrame:
        """
        Run full walk-forward validation
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            
        Returns:
            DataFrame with results for each fold
        """
        logger.info("=" * 70)
        logger.info("Starting Walk-Forward Validation")
        logger.info("=" * 70)
        
        # Get splits
        splits = self.splitter.split(df)
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        dates = df['date'].values
        
        results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            train_dates = dates[train_idx]
            test_dates = dates[test_idx]
            
            logger.info(f"Train: {train_dates[0]} to {train_dates[-1]} ({len(train_idx)} samples)")
            logger.info(f"Test:  {test_dates[0]} to {test_dates[-1]} ({len(test_idx)} samples)")
            
            # Train model
            model = self.train_baseline_model(X_train, y_train)
            
            # Predict on test set
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = self.evaluator.evaluate(y_test, y_pred_proba)
            self.evaluator.print_metrics(metrics, prefix="  ")
            
            # Store results
            fold_result = {
                'fold': fold_idx + 1,
                'train_start': str(train_dates[0]),
                'train_end': str(train_dates[-1]),
                'test_start': str(test_dates[0]),
                'test_end': str(test_dates[-1]),
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                **metrics
            }
            results.append(fold_result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        # Print summary
        self._print_summary()
        
        return results_df
    
    def _print_summary(self):
        """Print aggregate metrics across all folds"""
        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        metric_cols = ['auc', 'log_loss', 'brier_score', 'accuracy']
        
        print(f"\nAggregate Metrics (across {len(self.results)} folds):")
        print(f"{'Metric':<15} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 55)
        
        for metric in metric_cols:
            mean_val = self.results[metric].mean()
            std_val = self.results[metric].std()
            min_val = self.results[metric].min()
            max_val = self.results[metric].max()
            
            print(f"{metric:<15} {mean_val:<10.4f} {std_val:<10.4f} {min_val:<10.4f} {max_val:<10.4f}")
        
        logger.info("=" * 70)
    
    def plot_performance_over_time(self, save_path: str = "experiments/baseline_performance.png"):
        """Plot how metrics change over time"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Walk-Forward Validation - Performance Over Time', fontsize=16)
        
        metrics_to_plot = [
            ('auc', 'AUC (higher is better)'),
            ('log_loss', 'Log Loss (lower is better)'),
            ('brier_score', 'Brier Score (lower is better)'),
            ('accuracy', 'Accuracy (higher is better)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            x = np.arange(len(self.results))
            y = self.results[metric].values
            
            ax.plot(x, y, marker='o', linewidth=2, markersize=6)
            ax.axhline(y.mean(), color='r', linestyle='--', label=f'Mean: {y.mean():.4f}')
            ax.fill_between(x, y.mean() - y.std(), y.mean() + y.std(), alpha=0.2, color='r')
            
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance plot saved to {save_path}")
        plt.close()
    
    def save_results(self, results_path: str = "experiments/baseline_results.csv"):
        """Save detailed results to CSV"""
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    
    def train_champion_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'target',
        save_path: str = "experiments/champion_v0.pkl"
    ):
        """
        Train final champion model on all data
        
        Args:
            df: Full dataset
            feature_cols: Feature columns
            target_col: Target column
            save_path: Where to save the model
        """
        logger.info("\n" + "=" * 70)
        logger.info("Training Champion Model (v0)")
        logger.info("=" * 70)
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        model = self.train_baseline_model(X, y)
        
        # Save model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_type': 'LogisticRegression',
            'version': 'v0',
            'train_samples': len(X),
            'features': feature_cols,
            'trained_at': datetime.now().isoformat(),
            'validation_metrics': {
                'mean_auc': float(self.results['auc'].mean()),
                'mean_log_loss': float(self.results['log_loss'].mean()),
                'mean_brier_score': float(self.results['brier_score'].mean()),
                'mean_accuracy': float(self.results['accuracy'].mean())
            }
        }
        
        metadata_path = save_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Champion model saved to {save_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        logger.info("=" * 70)
        
        return model


def main():
    """Run baseline model with walk-forward validation"""
    
    # Load feature data
    logger.info("Loading feature data...")
    df = pd.read_csv("data/nvda_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    # Load feature names
    with open("data/feature_names.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Initialize validator
    validator = WalkForwardValidator(
        train_window=252,  # ~1 year
        test_window=21,    # ~1 month
        step_size=21       # ~1 month forward
    )
    
    # Run validation
    results_df = validator.run_validation(df, feature_cols)
    
    # Save results
    validator.save_results()
    
    # Plot performance
    validator.plot_performance_over_time()
    
    # Train and save champion model
    validator.train_champion_model(df, feature_cols)
    
    print("\n✅ Baseline model validation complete!")
    print("   - Results saved to experiments/baseline_results.csv")
    print("   - Performance plot saved to experiments/baseline_performance.png")
    print("   - Champion model saved to experiments/champion_v0.pkl")


if __name__ == "__main__":
    main()