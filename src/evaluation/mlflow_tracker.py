"""
MLflow Experiment Tracking & Model Registry
Logs parameters, metrics, artifacts, and registers models
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import pickle

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowExperimentTracker:
    """
    Manages MLflow experiment tracking and model registry
    """
    
    def __init__(
        self, 
        experiment_name: str = "stock-prediction",
        tracking_uri: str = "./mlruns"
    ):
        """
        Initialize MLflow tracking
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: Where to store MLflow data
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient(tracking_uri=tracking_uri)
    
    def log_walk_forward_run(
        self,
        model: Any,
        model_name: str,
        model_params: Dict[str, Any],
        feature_cols: List[str],
        results_df: pd.DataFrame,
        performance_plot_path: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log a complete walk-forward validation run
        
        Args:
            model: Trained model object
            model_name: Name/type of model (e.g., "LogisticRegression")
            model_params: Model hyperparameters
            feature_cols: List of feature names
            results_df: DataFrame with fold-by-fold results
            performance_plot_path: Path to performance plot
            tags: Additional tags for the run
            
        Returns:
            run_id: MLflow run ID
        """
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run: {run_id}")
            
            # Log parameters
            logger.info("Logging parameters...")
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("n_folds", len(results_df))
            mlflow.log_param("train_window", results_df['train_samples'].iloc[0])
            mlflow.log_param("test_window", results_df['test_samples'].iloc[0])
            
            # Log model hyperparameters
            for param_name, param_value in model_params.items():
                mlflow.log_param(f"model_{param_name}", param_value)
            
            # Log aggregate metrics
            logger.info("Logging metrics...")
            metric_cols = ['auc', 'log_loss', 'brier_score', 'accuracy']
            
            for metric in metric_cols:
                # Mean and std across folds
                mean_val = results_df[metric].mean()
                std_val = results_df[metric].std()
                min_val = results_df[metric].min()
                max_val = results_df[metric].max()
                
                mlflow.log_metric(f"{metric}_mean", mean_val)
                mlflow.log_metric(f"{metric}_std", std_val)
                mlflow.log_metric(f"{metric}_min", min_val)
                mlflow.log_metric(f"{metric}_max", max_val)
            
            # Log per-fold metrics for tracking over time
            for idx, row in results_df.iterrows():
                for metric in metric_cols:
                    mlflow.log_metric(f"fold_{metric}", row[metric], step=idx)
            
            # Log artifacts
            logger.info("Logging artifacts...")
            
            # Save and log feature list
            feature_path = Path("mlruns/temp_features.txt")
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            with open(feature_path, 'w') as f:
                f.write('\n'.join(feature_cols))
            mlflow.log_artifact(str(feature_path), artifact_path="features")
            
            # Save and log results CSV
            results_path = Path("mlruns/temp_results.csv")
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path), artifact_path="evaluation")
            
            # Log performance plot if provided
            if performance_plot_path and Path(performance_plot_path).exists():
                mlflow.log_artifact(performance_plot_path, artifact_path="plots")
            
            # Log model
            logger.info("Logging model...")
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=None  # We'll register separately
            )
            
            # Log tags
            if tags:
                mlflow.set_tags(tags)
            
            # Add default tags
            mlflow.set_tag("framework", "sklearn")
            mlflow.set_tag("validation_type", "walk_forward")
            mlflow.set_tag("timestamp", datetime.now().isoformat())
            
            logger.info(f"✓ Run logged successfully: {run_id}")
            
            return run_id
    
    def register_model(
        self,
        run_id: str,
        model_name: str = "nvda-predictor",
        description: Optional[str] = None
    ) -> str:
        """
        Register a model in MLflow Model Registry
        
        Args:
            run_id: MLflow run ID containing the model
            model_name: Name for the registered model
            description: Description of the model
            
        Returns:
            version: Model version number
        """
        logger.info(f"Registering model from run {run_id}...")
        
        model_uri = f"runs:/{run_id}/model"
        
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        version = result.version
        logger.info(f"✓ Model registered: {model_name} version {version}")
        
        # Add description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
        
        return version
    
    def tag_champion_model(
        self,
        model_name: str,
        version: str,
        stage: str = "Production"
    ):
        """
        Tag a model version as champion and transition to stage
        
        Args:
            model_name: Registered model name
            version: Model version
            stage: Stage to transition to (Staging/Production/Archived)
        """
        logger.info(f"Tagging {model_name} v{version} as champion...")
        
        # Transition to stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        # Add champion tag
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="champion",
            value="true"
        )
        
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="promoted_at",
            value=datetime.now().isoformat()
        )
        
        logger.info(f"✓ Model promoted to {stage} stage")
    
    def get_champion_model(self, model_name: str = "nvda-predictor"):
        """
        Load the current champion model from registry
        
        Args:
            model_name: Registered model name
            
        Returns:
            Loaded model object
        """
        logger.info(f"Loading champion model: {model_name}")
        
        try:
            # Try to get Production model
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("✓ Loaded model from Production stage")
            return model
        except Exception as e:
            logger.warning(f"No Production model found: {e}")
            
            # Fall back to latest version
            try:
                model_uri = f"models:/{model_name}/latest"
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("✓ Loaded latest model version")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    def compare_runs(
        self,
        run_id_1: str,
        run_id_2: str,
        metric: str = "auc_mean"
    ) -> Dict[str, Any]:
        """
        Compare two runs on a specific metric
        
        Args:
            run_id_1: First run ID (e.g., champion)
            run_id_2: Second run ID (e.g., challenger)
            metric: Metric to compare
            
        Returns:
            Comparison results
        """
        run1 = self.client.get_run(run_id_1)
        run2 = self.client.get_run(run_id_2)
        
        metric1 = run1.data.metrics.get(metric)
        metric2 = run2.data.metrics.get(metric)
        
        if metric1 is None or metric2 is None:
            raise ValueError(f"Metric {metric} not found in one or both runs")
        
        comparison = {
            'run_1': {
                'run_id': run_id_1,
                'value': metric1
            },
            'run_2': {
                'run_id': run_id_2,
                'value': metric2
            },
            'difference': metric2 - metric1,
            'improvement_pct': ((metric2 - metric1) / metric1) * 100 if metric1 != 0 else 0,
            'run_2_better': metric2 > metric1
        }
        
        return comparison
    
    def list_experiments(self) -> pd.DataFrame:
        """List all runs in the current experiment"""
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def get_best_run(self, metric: str = "metrics.auc_mean") -> str:
        """
        Get the run ID with the best value for a metric
        
        Args:
            metric: Metric to optimize (e.g., "metrics.auc_mean")
            
        Returns:
            run_id of best run
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"{metric} DESC"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found in experiment")
        
        best_run_id = runs.iloc[0]['run_id']
        best_value = runs.iloc[0][metric]
        
        logger.info(f"Best run: {best_run_id} with {metric}={best_value:.4f}")
        
        return best_run_id


def demo_mlflow_tracking():
    """
    Demo: Log a baseline model run with MLflow
    """
    logger.info("=" * 70)
    logger.info("MLflow Tracking Demo - Logging Baseline Model")
    logger.info("=" * 70)
    
    # Initialize tracker
    tracker = MLflowExperimentTracker(
        experiment_name="stock-prediction",
        tracking_uri="./mlruns"
    )
    
    # Load previous validation results
    logger.info("\nLoading baseline validation results...")
    results_df = pd.read_csv("experiments/baseline_results.csv")
    
    # Load champion model
    with open("experiments/champion_v0.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Load metadata
    with open("experiments/champion_v0_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['features']
    
    # Model parameters
    model_params = {
        'max_iter': 1000,
        'random_state': 42,
        'class_weight': 'balanced',
        'solver': 'lbfgs'
    }
    
    # Log the run
    run_id = tracker.log_walk_forward_run(
        model=model,
        model_name="LogisticRegression",
        model_params=model_params,
        feature_cols=feature_cols,
        results_df=results_df,
        performance_plot_path="experiments/baseline_performance.png",
        tags={
            "model_version": "v0",
            "baseline": "true",
            "champion": "true"
        }
    )
    
    # Register model
    version = tracker.register_model(
        run_id=run_id,
        model_name="nvda-predictor",
        description="Baseline logistic regression model - Champion v0"
    )
    
    # Tag as champion
    tracker.tag_champion_model(
        model_name="nvda-predictor",
        version=version,
        stage="Production"
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ MLflow Tracking Complete!")
    logger.info("=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Model: nvda-predictor v{version} (Production)")
    logger.info(f"\nView in MLflow UI:")
    logger.info(f"  cd {Path.cwd()}")
    logger.info(f"  mlflow ui")
    logger.info(f"  Then open: http://localhost:5000")
    logger.info("=" * 70)


if __name__ == "__main__":
    demo_mlflow_tracking()