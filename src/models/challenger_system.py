import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
import pickle

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Import our custom modules
import sys
sys.path.append('src')
from models.walk_forward_eval import WalkForwardValidator
from evaluation.mlflow_tracker import MLflowExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromotionRules:
    """
    Defines rules for promoting a challenger to champion
    """
    
    def __init__(
        self,
        auc_improvement_threshold: float = 0.01,  # 1% improvement
        log_loss_max_regression: float = 0.05,    # Max 5% worse
        brier_max_regression: float = 0.05        # Max 5% worse
    ):
        """
        Initialize promotion rules
        
        Args:
            auc_improvement_threshold: Minimum AUC improvement required
            log_loss_max_regression: Maximum allowed log loss increase (as fraction)
            brier_max_regression: Maximum allowed Brier score increase (as fraction)
        """
        self.auc_improvement_threshold = auc_improvement_threshold
        self.log_loss_max_regression = log_loss_max_regression
        self.brier_max_regression = brier_max_regression
    
    def evaluate_promotion(
        self,
        champion_metrics: Dict[str, float],
        challenger_metrics: Dict[str, float]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate whether challenger should be promoted
        
        Args:
            champion_metrics: Champion model metrics
            challenger_metrics: Challenger model metrics
            
        Returns:
            (should_promote, reason, details)
        """
        details = {
            'checks': {},
            'metrics_comparison': {}
        }
        
        # Extract mean metrics
        champ_auc = champion_metrics.get('auc_mean', 0)
        chall_auc = challenger_metrics.get('auc_mean', 0)
        
        champ_log_loss = champion_metrics.get('log_loss_mean', float('inf'))
        chall_log_loss = challenger_metrics.get('log_loss_mean', float('inf'))
        
        champ_brier = champion_metrics.get('brier_score_mean', float('inf'))
        chall_brier = challenger_metrics.get('brier_score_mean', float('inf'))
        
        # Store comparisons
        details['metrics_comparison'] = {
            'auc': {
                'champion': champ_auc,
                'challenger': chall_auc,
                'improvement': chall_auc - champ_auc,
                'improvement_pct': ((chall_auc - champ_auc) / champ_auc * 100) if champ_auc > 0 else 0
            },
            'log_loss': {
                'champion': champ_log_loss,
                'challenger': chall_log_loss,
                'change': chall_log_loss - champ_log_loss,
                'change_pct': ((chall_log_loss - champ_log_loss) / champ_log_loss * 100) if champ_log_loss > 0 else 0
            },
            'brier_score': {
                'champion': champ_brier,
                'challenger': chall_brier,
                'change': chall_brier - champ_brier,
                'change_pct': ((chall_brier - champ_brier) / champ_brier * 100) if champ_brier > 0 else 0
            }
        }
        
        # Check 1: AUC Improvement
        auc_improvement = chall_auc - champ_auc
        auc_check = auc_improvement >= self.auc_improvement_threshold
        details['checks']['auc_improvement'] = {
            'passed': auc_check,
            'improvement': auc_improvement,
            'threshold': self.auc_improvement_threshold
        }
        
        # Check 2: Log Loss Non-Regression
        log_loss_regression = (chall_log_loss - champ_log_loss) / champ_log_loss if champ_log_loss > 0 else 0
        log_loss_check = log_loss_regression <= self.log_loss_max_regression
        details['checks']['log_loss_non_regression'] = {
            'passed': log_loss_check,
            'regression_pct': log_loss_regression * 100,
            'max_allowed_pct': self.log_loss_max_regression * 100
        }
        
        # Check 3: Brier Score Non-Regression
        brier_regression = (chall_brier - champ_brier) / champ_brier if champ_brier > 0 else 0
        brier_check = brier_regression <= self.brier_max_regression
        details['checks']['brier_non_regression'] = {
            'passed': brier_check,
            'regression_pct': brier_regression * 100,
            'max_allowed_pct': self.brier_max_regression * 100
        }
        
        # Final decision
        all_checks_passed = auc_check and log_loss_check and brier_check
        
        if all_checks_passed:
            reason = f"✓ Promotion approved: AUC improved by {auc_improvement:.4f}, no significant regression in log loss or Brier score"
        else:
            failed_checks = []
            if not auc_check:
                failed_checks.append(f"AUC improvement ({auc_improvement:.4f}) below threshold ({self.auc_improvement_threshold})")
            if not log_loss_check:
                failed_checks.append(f"Log loss regressed by {log_loss_regression*100:.2f}% (max allowed: {self.log_loss_max_regression*100:.2f}%)")
            if not brier_check:
                failed_checks.append(f"Brier score regressed by {brier_regression*100:.2f}% (max allowed: {self.brier_max_regression*100:.2f}%)")
            
            reason = "✗ Promotion rejected: " + "; ".join(failed_checks)
        
        return all_checks_passed, reason, details


class ChallengerTrainer:
    """
    Trains challenger models and compares against champion
    """
    
    def __init__(self):
        self.validator = WalkForwardValidator(
            train_window=252,
            test_window=21,
            step_size=21
        )
        self.tracker = MLflowExperimentTracker()
        self.promotion_rules = PromotionRules()
    
    def get_challenger_configs(self) -> List[Dict[str, Any]]:
        """
        Define challenger model configurations
        
        Returns:
            List of model configs to try
        """
        configs = [
            {
                'name': 'LogisticRegression_L1',
                'model_class': LogisticRegression,
                'params': {
                    'max_iter': 1000,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'penalty': 'l1',
                    'solver': 'saga',
                    'C': 0.1
                },
                'description': 'Logistic Regression with L1 regularization'
            },
            {
                'name': 'RandomForest',
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 5,
                    'min_samples_split': 20,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'n_jobs': -1
                },
                'description': 'Random Forest with limited depth to prevent overfitting'
            },
            {
                'name': 'LightGBM',
                'model_class': LGBMClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.05,
                    'num_leaves': 15,
                    'random_state': 42,
                    'class_weight': 'balanced',
                    'verbose': -1
                },
                'description': 'LightGBM with conservative parameters'
            }
        ]
        
        return configs
    
    def train_challenger(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        config: Dict[str, Any],
        target_col: str = 'target'
    ) -> Tuple[Any, pd.DataFrame, str]:
        """
        Train a challenger model with walk-forward validation
        
        Args:
            df: Feature dataframe
            feature_cols: List of feature columns
            config: Model configuration
            target_col: Target column name
            
        Returns:
            (model, results_df, run_id)
        """
        logger.info("=" * 70)
        logger.info(f"Training Challenger: {config['name']}")
        logger.info("=" * 70)
        
        # Modify validator to use challenger model
        original_train_method = self.validator.train_baseline_model
        
        def train_challenger_model(X_train, y_train):
            model = config['model_class'](**config['params'])
            model.fit(X_train, y_train)
            return model
        
        self.validator.train_baseline_model = train_challenger_model
        
        # Run walk-forward validation
        results_df = self.validator.run_validation(df, feature_cols, target_col)
        
        # Train final model on all data
        X = df[feature_cols].values
        y = df[target_col].values
        final_model = train_challenger_model(X, y)
        
        # Plot performance
        plot_path = f"experiments/challenger_{config['name']}_performance.png"
        self.validator.plot_performance_over_time(save_path=plot_path)
        
        # Log to MLflow
        run_id = self.tracker.log_walk_forward_run(
            model=final_model,
            model_name=config['name'],
            model_params=config['params'],
            feature_cols=feature_cols,
            results_df=results_df,
            performance_plot_path=plot_path,
            tags={
                'model_type': 'challenger',
                'description': config['description']
            }
        )
        
        # Restore original method
        self.validator.train_baseline_model = original_train_method
        
        logger.info(f"✓ Challenger training complete. Run ID: {run_id}")
        
        return final_model, results_df, run_id
    
    def compare_and_promote(
        self,
        champion_run_id: str,
        challenger_run_id: str,
        challenger_name: str,
        auto_promote: bool = False
    ) -> Dict[str, Any]:
        """
        Compare challenger against champion and decide on promotion
        
        Args:
            champion_run_id: MLflow run ID of champion
            challenger_run_id: MLflow run ID of challenger
            challenger_name: Name of challenger model
            auto_promote: Whether to automatically promote if rules pass
            
        Returns:
            Promotion decision details
        """
        logger.info("=" * 70)
        logger.info("Champion vs Challenger Comparison")
        logger.info("=" * 70)
        
        # Get metrics from both runs
        client = self.tracker.client
        
        champion_run = client.get_run(champion_run_id)
        challenger_run = client.get_run(challenger_run_id)
        
        champion_metrics = champion_run.data.metrics
        challenger_metrics = challenger_run.data.metrics
        
        # Evaluate promotion
        should_promote, reason, details = self.promotion_rules.evaluate_promotion(
            champion_metrics,
            challenger_metrics
        )
        
        # Print comparison
        logger.info("\nMetrics Comparison:")
        logger.info("-" * 70)
        
        for metric_name, comparison in details['metrics_comparison'].items():
            logger.info(f"\n{metric_name.upper()}:")
            logger.info(f"  Champion:   {comparison['champion']:.4f}")
            logger.info(f"  Challenger: {comparison['challenger']:.4f}")
            
            if 'improvement' in comparison:
                logger.info(f"  Improvement: {comparison['improvement']:+.4f} ({comparison['improvement_pct']:+.2f}%)")
            else:
                logger.info(f"  Change: {comparison['change']:+.4f} ({comparison['change_pct']:+.2f}%)")
        
        logger.info("\n" + "-" * 70)
        logger.info("Promotion Checks:")
        logger.info("-" * 70)
        
        for check_name, check_details in details['checks'].items():
            status = "✓ PASS" if check_details['passed'] else "✗ FAIL"
            logger.info(f"{status} - {check_name}")
            for key, value in check_details.items():
                if key != 'passed':
                    logger.info(f"      {key}: {value}")
        
        logger.info("\n" + "=" * 70)
        logger.info(reason)
        logger.info("=" * 70)
        
        # Promotion decision
        decision = {
            'should_promote': should_promote,
            'reason': reason,
            'details': details,
            'champion_run_id': champion_run_id,
            'challenger_run_id': challenger_run_id,
            'challenger_name': challenger_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save decision log
        decision_path = Path("experiments/promotion_decisions.jsonl")
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(decision_path, 'a') as f:
            f.write(json.dumps(decision) + '\n')
        
        logger.info(f"Decision logged to {decision_path}")
        
        # Auto-promote if enabled and rules pass
        if auto_promote and should_promote:
            logger.info("\n🚀 Auto-promoting challenger to champion...")
            
            # Register new model
            version = self.tracker.register_model(
                run_id=challenger_run_id,
                model_name="nvda-predictor",
                description=f"Promoted challenger: {challenger_name}"
            )
            
            # Tag as champion
            self.tracker.tag_champion_model(
                model_name="nvda-predictor",
                version=version,
                stage="Production"
            )
            
            decision['promoted'] = True
            decision['new_version'] = version
            
            logger.info(f"✓ Challenger promoted to Production (version {version})")
        elif should_promote:
            logger.info("\n⚠️  Promotion approved but auto_promote=False. Manual promotion required.")
            decision['promoted'] = False
        else:
            logger.info("\n✗ Challenger rejected. Champion remains in production.")
            decision['promoted'] = False
        
        return decision


def main():
    """
    Demo: Train challengers and compare against champion
    """
    logger.info("=" * 70)
    logger.info("CHALLENGER TRAINING & PROMOTION PIPELINE")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nLoading feature data...")
    df = pd.read_csv("data/nvda_features.csv")
    df['date'] = pd.to_datetime(df['date'])
    
    with open("data/feature_names.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    logger.info(f"Loaded {len(df)} samples with {len(feature_cols)} features")
    
    # Initialize trainer
    trainer = ChallengerTrainer()
    
    # Get champion run ID (from previous MLflow run)
    tracker = MLflowExperimentTracker()
    champion_run_id = tracker.get_best_run(metric="metrics.auc_mean")
    logger.info(f"\nChampion run ID: {champion_run_id}")
    
    # Get challenger configs
    challenger_configs = trainer.get_challenger_configs()
    
    # Train each challenger
    results = []
    
    for config in challenger_configs:
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Challenger: {config['name']}")
        logger.info(f"{'='*70}")
        
        model, results_df, run_id = trainer.train_challenger(
            df=df,
            feature_cols=feature_cols,
            config=config
        )
        
        # Compare against champion
        decision = trainer.compare_and_promote(
            champion_run_id=champion_run_id,
            challenger_run_id=run_id,
            challenger_name=config['name'],
            auto_promote=False  # Set to True for automatic promotion
        )
        
        results.append({
            'challenger_name': config['name'],
            'run_id': run_id,
            'should_promote': decision['should_promote'],
            'reason': decision['reason']
        })
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CHALLENGER EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    for result in results:
        status = "✓ APPROVED" if result['should_promote'] else "✗ REJECTED"
        logger.info(f"\n{status}: {result['challenger_name']}")
        logger.info(f"  Run ID: {result['run_id']}")
        logger.info(f"  {result['reason']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ Challenger training complete!")
    logger.info("View results in MLflow UI: mlflow ui")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()