import json
import os
import tempfile
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict

import matplotlib.pyplot as plt

# --- MLflow Integration ---
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns

# --- LangGraph Import ---
from langgraph.graph import END, StateGraph
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# --- Conformal Prediction Import ---
try:
    from mapie.classification import MapieClassifier
    from mapie.metrics import (
        classification_coverage_score,
        classification_mean_width_score,
    )

    HAS_MAPIE = True
except ImportError:
    HAS_MAPIE = False
    print("Mapie not available - install with: pip install mapie")

# --- Optional Imports with Fallbacks ---
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real

    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print(
        "scikit-optimize not available, tuning will be limited. Install with: pip install scikit-optimize"
    )

try:
    import h2o
    from h2o.automl import H2OAutoML

    HAS_H2O = True
except ImportError:
    HAS_H2O = False
    print("H2O not available - install with: pip install h2o")

warnings.filterwarnings("ignore")


# --- MLflow Configuration ---
class MLflowManager:
    """Manages MLflow experiment tracking and logging."""

    def __init__(self, experiment_name: str = "ML_Pipeline_Experiment"):
        self.experiment_name = experiment_name
        # Set the tracking URI to SQLite
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        # Create the experiment if it doesn't exist
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        self.client = MlflowClient()
        self.parent_run_id = None

    def _get_or_create_experiment(self) -> str:
        """Get existing experiment or create new one."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
        except:
            pass
        return mlflow.create_experiment(self.experiment_name)

    def start_parent_run(self, run_name: str = "Pipeline_Run") -> str:
        """Start a parent run for the entire pipeline."""
        run = self.client.create_run(
            experiment_id=self.experiment_id,
            tags={
                "pipeline_type": "langgraph_ml_pipeline",
                "run_name": run_name,
                "timestamp": str(pd.Timestamp.now()),
            },
        )
        self.parent_run_id = run.info.run_id
        mlflow.start_run(run_id=self.parent_run_id)
        return self.parent_run_id

    def log_model_result(
        self,
        model_name: str,
        model_obj,
        metrics: Dict,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        stage: str = "training",
    ):
        """Log model, metrics, and artifacts as a nested run."""
        if not self.parent_run_id:
            raise ValueError("Parent run not started. Call start_parent_run() first.")

        with mlflow.start_run(run_id=self.parent_run_id, nested=True):
            with mlflow.start_run(
                nested=True, run_name=f"{model_name}_{stage}"
            ) as nested_run:
                # Log metrics with prefixes for better organization
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{stage}_{metric_name}", value)

                # Log model parameters with better organization
                if hasattr(model_obj, "get_params"):
                    params = model_obj.get_params()
                    # Filter and organize parameters
                    simple_params = {
                        k: v
                        for k, v in params.items()
                        if isinstance(v, (int, float, str, bool, type(None)))
                    }
                    # Group parameters by type
                    param_groups = {
                        "model_params": {},
                        "training_params": {},
                        "other_params": {},
                    }

                    for k, v in simple_params.items():
                        if any(
                            x in k.lower()
                            for x in ["n_estimators", "max_depth", "learning_rate"]
                        ):
                            param_groups["model_params"][k] = v
                        elif any(
                            x in k.lower()
                            for x in ["random_state", "n_jobs", "verbose"]
                        ):
                            param_groups["training_params"][k] = v
                        else:
                            param_groups["other_params"][k] = v

                    # Log grouped parameters
                    for group, params in param_groups.items():
                        if params:
                            mlflow.log_params(
                                {f"{group}.{k}": v for k, v in params.items()}
                            )

                # Log model based on type
                self._log_model_by_type(model_obj, model_name)

                # Enhanced tags for better organization
                mlflow.set_tags(
                    {
                        "model_type": type(model_obj).__name__,
                        "stage": stage,
                        "pipeline_step": stage,
                        "model_name": model_name,
                        "timestamp": str(pd.Timestamp.now()),
                    }
                )

                # Create and log enhanced visualizations
                self._log_enhanced_visualizations(y_true, y_prob, model_name, stage)

                return nested_run.info.run_id

    def _log_model_by_type(self, model_obj, model_name: str):
        """Log model based on its type."""
        try:
            if "XGB" in model_name and HAS_XGBOOST:
                mlflow.xgboost.log_model(
                    model_obj, f"model_{model_name.replace(' ', '_')}"
                )
            else:
                mlflow.sklearn.log_model(
                    model_obj, f"model_{model_name.replace(' ', '_')}"
                )
        except Exception as e:
            print(f"Warning: Could not log model {model_name}: {e}")
            # Try to log as pickle if sklearn logging fails
            try:
                mlflow.sklearn.log_model(
                    model_obj, f"model_{model_name.replace(' ', '_')}"
                )
            except:
                print(f"Failed to log model {model_name} entirely")

    def _log_enhanced_visualizations(
        self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str, stage: str
    ):
        """Create and log enhanced visualizations for model evaluation."""
        try:
            # Create a figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            axes[0, 0].plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
            axes[0, 0].plot([0, 1], [0, 1], "k--")
            axes[0, 0].set_xlabel("False Positive Rate")
            axes[0, 0].set_ylabel("True Positive Rate")
            axes[0, 0].set_title(f"ROC Curve - {model_name}")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            axes[0, 1].plot(recall, precision)
            axes[0, 1].set_xlabel("Recall")
            axes[0, 1].set_ylabel("Precision")
            axes[0, 1].set_title(f"PR Curve - {model_name}")
            axes[0, 1].grid(True)

            # Calibration Plot
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            axes[1, 0].plot(prob_pred, prob_true, marker="o", label="Model")
            axes[1, 0].plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
            axes[1, 0].set_xlabel("Mean Predicted Probability")
            axes[1, 0].set_ylabel("True Probability")
            axes[1, 0].set_title("Calibration Plot")
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            # Distribution of Predicted Probabilities
            sns.histplot(y_prob, bins=20, ax=axes[1, 1])
            axes[1, 1].set_xlabel("Predicted Probability")
            axes[1, 1].set_ylabel("Count")
            axes[1, 1].set_title("Distribution of Predictions")
            axes[1, 1].grid(True)

            plt.tight_layout()

            # Save to temporary file and log
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
                mlflow.log_artifact(
                    tmp.name,
                    f"plots/{model_name.replace(' ', '_')}_{stage}_analysis.png",
                )
                os.unlink(tmp.name)

            plt.close()

        except Exception as e:
            print(
                f"Warning: Could not create enhanced visualizations for {model_name}: {e}"
            )

    def log_dataset_info(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        X_calib: pd.DataFrame,
        target: str,
    ):
        """Log enhanced dataset information."""
        if not self.parent_run_id:
            return

        with mlflow.start_run(run_id=self.parent_run_id, nested=True):
            # Basic dataset info
            mlflow.log_params(
                {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "calibration_samples": len(X_calib),
                    "n_features": X_train.shape[1] - 1,  # Exclude target
                    "target_column": target,
                    "feature_columns": list(X_train.columns.drop(target))[
                        :10
                    ],  # Log first 10 features
                }
            )

            # Enhanced class distribution logging
            train_dist = X_train[target].value_counts(normalize=True).to_dict()
            test_dist = X_test[target].value_counts(normalize=True).to_dict()
            calib_dist = X_calib[target].value_counts(normalize=True).to_dict()

            for split, dist in [
                ("train", train_dist),
                ("test", test_dist),
                ("calib", calib_dist),
            ]:
                for class_val, proportion in dist.items():
                    mlflow.log_metric(
                        f"{split}_class_{class_val}_proportion", proportion
                    )

            # Log feature statistics with valid metric names
            feature_stats = X_train.describe().to_dict()
            for feature, stats in feature_stats.items():
                if feature != target:
                    for stat_name, value in stats.items():
                        # Replace special characters in stat_name
                        safe_stat_name = stat_name.replace("%", "pct").replace(" ", "_")
                        mlflow.log_metric(f"feature_{feature}_{safe_stat_name}", value)

            # Create and log feature distribution plots
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                # Target distribution
                sns.countplot(data=X_train, x=target, ax=axes[0, 0])
                axes[0, 0].set_title("Target Distribution")

                # Feature correlations
                corr_matrix = X_train.corr()
                sns.heatmap(corr_matrix, ax=axes[0, 1], cmap="coolwarm")
                axes[0, 1].set_title("Feature Correlations")

                # Feature distributions
                for i, feature in enumerate(X_train.columns[:2]):
                    if feature != target:
                        sns.histplot(data=X_train, x=feature, ax=axes[1, i])
                        axes[1, i].set_title(f"{feature} Distribution")

                plt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
                    mlflow.log_artifact(tmp.name, "plots/dataset_analysis.png")
                    os.unlink(tmp.name)

                plt.close()
            except Exception as e:
                print(f"Warning: Could not create dataset visualizations: {e}")

    def log_pipeline_decision(
        self, decision: str, best_model_name: str, best_metrics: Dict
    ):
        """Log enhanced pipeline decision and best model info."""
        if not self.parent_run_id:
            return

        with mlflow.start_run(run_id=self.parent_run_id, nested=True):
            # Log decision details
            mlflow.log_params(
                {
                    "pipeline_decision": decision,
                    "best_model": best_model_name,
                    "decision_timestamp": str(pd.Timestamp.now()),
                }
            )

            # Log best model metrics with enhanced organization
            for metric_name, value in best_metrics.items():
                mlflow.log_metric(f"best_model_{metric_name}", value)

            # Create and log decision visualization
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics_df = pd.DataFrame(
                    {
                        "Metric": list(best_metrics.keys()),
                        "Value": list(best_metrics.values()),
                    }
                )
                sns.barplot(data=metrics_df, x="Metric", y="Value", ax=ax)
                ax.set_title(f"Best Model Metrics ({best_model_name})")
                plt.xticks(rotation=45)
                plt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
                    mlflow.log_artifact(tmp.name, "plots/best_model_metrics.png")
                    os.unlink(tmp.name)

                plt.close()
            except Exception as e:
                print(f"Warning: Could not create decision visualization: {e}")

    def log_final_comparison(self, results_df: pd.DataFrame):
        """Log enhanced final model comparison results."""
        if not self.parent_run_id:
            return

        with mlflow.start_run(run_id=self.parent_run_id, nested=True):
            # Save results table as artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                results_df.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, "results/model_comparison.csv")
                os.unlink(tmp.name)

            # Create and log comparison visualizations
            try:
                # Model comparison plot
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))

                # AUC comparison
                sns.barplot(data=results_df, x="Model", y="auc_score", ax=axes[0, 0])
                axes[0, 0].set_title("AUC Score Comparison")
                plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

                # Brier score comparison
                sns.barplot(data=results_df, x="Model", y="brier_score", ax=axes[0, 1])
                axes[0, 1].set_title("Brier Score Comparison")
                plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)

                # F1 score comparison
                sns.barplot(data=results_df, x="Model", y="f1_score", ax=axes[1, 0])
                axes[1, 0].set_title("F1 Score Comparison")
                plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)

                # Precision-Recall comparison
                sns.scatterplot(
                    data=results_df,
                    x="precision",
                    y="recall",
                    hue="Model",
                    ax=axes[1, 1],
                )
                axes[1, 1].set_title("Precision-Recall Trade-off")

                plt.tight_layout()

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    plt.savefig(tmp.name, dpi=150, bbox_inches="tight")
                    mlflow.log_artifact(tmp.name, "plots/model_comparison.png")
                    os.unlink(tmp.name)

                plt.close()
            except Exception as e:
                print(f"Warning: Could not create comparison visualizations: {e}")

    def finish_run(self):
        """End the parent run with enhanced logging."""
        if self.parent_run_id:
            mlflow.end_run()
            print("\nMLflow run ended successfully.")


# --- State and Result Management ---
class ModelResults:
    """Class to store and manage model results for before/after comparison."""

    def __init__(self, mlflow_manager: Optional[MLflowManager] = None):
        self.results = []
        self.before_mapie_results = []
        self.mlflow_manager = mlflow_manager

    def add_result(self, model_name: str, metrics: Dict[str, Any], **kwargs):
        """Add or update a result to the main list."""
        self.results = [r for r in self.results if r["model_name"] != model_name]
        result_entry = {"model_name": model_name, "metrics": metrics, **kwargs}
        self.results.append(result_entry)

        # Log to MLflow
        if (
            self.mlflow_manager
            and "model_object" in kwargs
            and "y_true" in kwargs
            and "y_prob" in kwargs
        ):
            stage = (
                "post_processing"
                if any(x in model_name for x in ["Tuned", "Calibrated", "MAPIE"])
                else "training"
            )
            self.mlflow_manager.log_model_result(
                model_name,
                kwargs["model_object"],
                metrics,
                kwargs["y_true"],
                kwargs["y_prob"],
                stage,
            )

    def add_before_mapie_result(
        self, model_name: str, metrics: Dict[str, Any], **kwargs
    ):
        """Add a result to the pre-Mapie snapshot."""
        self.before_mapie_results = [
            r for r in self.before_mapie_results if r["model_name"] != model_name
        ]
        result_entry = {"model_name": model_name, "metrics": metrics, **kwargs}
        self.before_mapie_results.append(result_entry)

    def get_results_df(self, before_mapie: bool = False) -> pd.DataFrame:
        """Get results as a sorted DataFrame."""
        results_to_use = self.before_mapie_results if before_mapie else self.results
        if not results_to_use:
            return pd.DataFrame()

        df_data = [
            {"Model": r["model_name"], **r["metrics"]}
            for r in results_to_use
            if "error" not in r["metrics"]
        ]
        df = pd.DataFrame(df_data)

        if "auc_score" in df.columns:
            return df.sort_values(by="auc_score", ascending=False).reset_index(
                drop=True
            )
        return df

    def get_best_model_result(self, metric: str = "auc_score") -> Optional[Dict]:
        """Get the full result dictionary for the best performing model from the main list."""
        df = self.get_results_df()
        if df.empty or metric not in df.columns:
            return None

        non_mapie_df = df[~df["Model"].str.contains("(With MAPIE)")]
        if not non_mapie_df.empty:
            best_model_name = non_mapie_df.iloc[0]["Model"]
        else:
            best_model_name = df.iloc[0]["Model"]

        return next(
            (r for r in self.results if r["model_name"] == best_model_name), None
        )


class MLPipelineState(TypedDict):
    """State definition for the ML pipeline graph."""

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    x_calibrate: pd.DataFrame
    target: str
    model_results: ModelResults
    best_model_info: Optional[Dict]
    decision: str
    mlflow_manager: Optional[MLflowManager]


# --- Metric Calculation ---
def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive metrics for binary classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_score": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


# --- LangGraph Nodes ---
def initial_model_training_node(state: MLPipelineState) -> Dict[str, Any]:
    """Trains base models, stores their results, then applies Mapie and stores results again."""
    print("=" * 60 + "\nNODE: Initial Model Training & Conformalization\n" + "=" * 60)
    target = state["target"]
    results_obj = state["model_results"]
    mlflow_manager = state.get("mlflow_manager")
    feature_cols = [col for col in state["x_train"].columns if col != target]

    X_train, y_train = state["x_train"][feature_cols], state["x_train"][target]
    X_test, y_test = state["x_test"][feature_cols], state["x_test"][target]
    X_calib, y_calib = state["x_calibrate"][feature_cols], state["x_calibrate"][target]

    # Log dataset info to MLflow
    if mlflow_manager:
        mlflow_manager.log_dataset_info(
            state["x_train"], state["x_test"], state["x_calibrate"], target
        )

    model_configs = [
        {
            "name": "Logistic Regression",
            "model": LogisticRegression(random_state=42, max_iter=1000),
        },
        {"name": "Random Forest", "model": RandomForestClassifier(random_state=42)},
        *(
            [
                {
                    "name": "XGBoost",
                    "model": xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
                }
            ]
            if HAS_XGBOOST
            else []
        ),
    ]

    trained_models_for_mapie = {}

    print("\n--- Phase 1: Training Base Models (Before MAPIE) ---")
    for config in model_configs:
        print(f"Training {config['name']}...")
        try:
            model = config["model"].fit(X_train, y_train)
            trained_models_for_mapie[config["name"]] = model
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = compute_all_metrics(y_test.values, y_pred, y_prob)

            print(
                f"  ✓ {config['name']} (Before) | AUC: {metrics.get('auc_score', 0):.4f}"
            )

            # Store results for "before" comparison
            results_obj.add_before_mapie_result(
                config["name"],
                metrics,
                model_object=model,
                y_true=y_test.values,
                y_prob=y_prob,
            )
            # Also add to the main results list initially
            results_obj.add_result(
                config["name"],
                metrics,
                model_object=model,
                y_true=y_test.values,
                y_prob=y_prob,
            )
        except Exception as e:
            print(f"  ✗ {config['name']} failed: {e}")

    if HAS_MAPIE:
        print("\n--- Phase 2: Applying Conformal Prediction (MAPIE) ---")
        for name, model in trained_models_for_mapie.items():
            print(f"Applying MAPIE to {name}...")
            try:
                mapie = MapieClassifier(estimator=model, cv="prefit", method="score")
                mapie.fit(X_calib, y_calib)
                y_pred_mapie, y_ps = mapie.predict(X_test, alpha=0.1)

                original_y_prob = next(
                    r["y_prob"]
                    for r in results_obj.before_mapie_results
                    if r["model_name"] == name
                )

                metrics_mapie = compute_all_metrics(
                    y_test.values, y_pred_mapie, original_y_prob
                )
                metrics_mapie["conformal_coverage"] = classification_coverage_score(
                    y_test.values, np.squeeze(y_ps)
                )
                metrics_mapie["avg_set_size"] = classification_mean_width_score(
                    np.squeeze(y_ps)
                )

                print(
                    f"  ✓ {name} (With MAPIE) | Coverage: {metrics_mapie.get('conformal_coverage', 0):.4f}"
                )
                results_obj.add_result(
                    f"{name} (With MAPIE)",
                    metrics_mapie,
                    model_object=mapie,
                    y_true=y_test.values,
                    y_prob=original_y_prob,
                )
            except Exception as e:
                print(f"  ✗ MAPIE application to {name} failed: {e}")

    return {"model_results": results_obj}


def evaluate_and_decide_node(state: MLPipelineState) -> Dict[str, Any]:
    """Heuristically decides the next step: tune, calibrate, or report."""
    print("=" * 60 + "\nNODE: Evaluate & Decide\n" + "=" * 60)
    results = state["model_results"]
    mlflow_manager = state.get("mlflow_manager")

    best_model_result = results.get_best_model_result("auc_score")

    if not best_model_result or "H2O" in best_model_result["model_name"]:
        return {"decision": "proceed_to_report"}

    metrics = best_model_result["metrics"]
    model_name = best_model_result["model_name"]
    print(
        f"  > Evaluating best model: {model_name} (AUC: {metrics.get('auc_score', 0):.4f})"
    )

    AUC_LOW_THRESHOLD, BRIER_HIGH_THRESHOLD = 0.75, 0.18
    decision = "proceed_to_report"
    if metrics.get("auc_score", 0) < AUC_LOW_THRESHOLD:
        decision = "tune_best_model"
        print(f"  ! Decision: AUC is low. Triggering tuning.")
    elif metrics.get("brier_score", 0) > BRIER_HIGH_THRESHOLD:
        decision = "calibrate_best_model"
        print(f"  ! Decision: Brier Score is high. Triggering calibration.")
    else:
        print("  ✓ Decision: Performance acceptable. Proceeding to report.")

    # Log decision to MLflow
    if mlflow_manager:
        mlflow_manager.log_pipeline_decision(decision, model_name, metrics)

    return {"decision": decision, "best_model_info": best_model_result}


def hyperparameter_tuning_node(state: MLPipelineState) -> Dict[str, Any]:
    """Performs hyperparameter tuning on the best model."""
    print("=" * 60 + "\nNODE: Hyperparameter Tuning\n" + "=" * 60)
    if not HAS_SKOPT or not state.get("best_model_info"):
        return {}

    model_info = state["best_model_info"]
    model_name = model_info["model_name"].replace(" (With MAPIE)", "")
    print(f"  > Tuning: {model_name}")

    X_train, y_train = (
        state["x_train"].drop(columns=[state["target"]]),
        state["x_train"][state["target"]],
    )
    X_test, y_test = (
        state["x_test"].drop(columns=[state["target"]]),
        state["x_test"][state["target"]],
    )

    param_spaces = {
        "Random Forest": {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(10, 50),
        },
        "XGBoost": {"n_estimators": Integer(100, 500), "max_depth": Integer(3, 10)},
        "Logistic Regression": {"C": Real(1e-6, 1e6, "log-uniform")},
    }
    if model_name not in param_spaces:
        return {}

    opt = BayesSearchCV(
        model_info["model_object"],
        param_spaces[model_name],
        n_iter=32,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
    )
    opt.fit(X_train, y_train)

    tuned_model = opt.best_estimator_
    y_pred = tuned_model.predict(X_test)
    y_prob = tuned_model.predict_proba(X_test)[:, 1]
    tuned_metrics = compute_all_metrics(y_test.values, y_pred, y_prob)

    print(f"  ✓ {model_name} (Tuned) | AUC: {tuned_metrics.get('auc_score', 0):.4f}")
    state["model_results"].add_result(
        f"{model_name} (Tuned)",
        tuned_metrics,
        model_object=tuned_model,
        y_true=y_test.values,
        y_prob=y_prob,
    )
    return {"model_results": state["model_results"]}


def calibrate_model_node(state: MLPipelineState) -> Dict[str, Any]:
    """Calibrates the best model's probabilities."""
    print("=" * 60 + "\nNODE: Model Calibration\n" + "=" * 60)
    if not state.get("best_model_info"):
        return {}

    model_info = state["best_model_info"]
    model_name = model_info["model_name"].replace(" (With MAPIE)", "")
    print(f"  > Calibrating: {model_name}")

    X_calib, y_calib = (
        state["x_calibrate"].drop(columns=[state["target"]]),
        state["x_calibrate"][state["target"]],
    )
    X_test, y_test = (
        state["x_test"].drop(columns=[state["target"]]),
        state["x_test"][state["target"]],
    )

    calibrated_model = CalibratedClassifierCV(
        model_info["model_object"], cv="prefit", method="isotonic"
    )
    calibrated_model.fit(X_calib, y_calib)

    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    calib_metrics = compute_all_metrics(y_test.values, y_pred, y_prob)

    print(
        f"  ✓ {model_name} (Calibrated) | Brier: {calib_metrics.get('brier_score', 0):.4f}"
    )
    state["model_results"].add_result(
        f"{model_name} (Calibrated)",
        calib_metrics,
        model_object=calibrated_model,
        y_true=y_test.values,
        y_prob=y_prob,
    )
    return {"model_results": state["model_results"]}


def final_report_node(state: MLPipelineState) -> Dict[str, Any]:
    """Generates a PDF report with metrics and per-model comparison graphs."""
    print("=" * 60 + "\nNODE: Generate Final Report\n" + "=" * 60)
    results_obj = state["model_results"]
    mlflow_manager = state.get("mlflow_manager")
    output_path = "model_comparison_report_v2.pdf"

    before_df = results_obj.get_results_df(before_mapie=True)
    after_df = results_obj.get_results_df(before_mapie=False)

    if after_df.empty:
        print("  ! No models to report. Skipping PDF generation.")
        return {}

    # Log final comparison to MLflow
    if mlflow_manager:
        with mlflow.start_run(run_id=mlflow_manager.parent_run_id, nested=True):
            # Save results table as artifact
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".csv", delete=False
            ) as tmp:
                after_df.to_csv(tmp.name, index=False)
                mlflow.log_artifact(tmp.name, "results/model_comparison.csv")
                os.unlink(tmp.name)

    with PdfPages(output_path) as pdf:
        plt.style.use("seaborn-v0_8-whitegrid")

        # Page 1: Combined Metrics Table
        fig, ax = plt.subplots(figsize=(14, max(4, len(after_df) * 0.5)))
        ax.axis("tight")
        ax.axis("off")
        display_cols = [
            c
            for c in [
                "Model",
                "auc_score",
                "brier_score",
                "f1_score",
                "conformal_coverage",
                "avg_set_size",
            ]
            if c in after_df.columns
        ]
        table = ax.table(
            cellText=after_df[display_cols].round(4).values,
            colLabels=display_cols,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        # Set font size for all cells
        for cell in table._cells:
            table._cells[cell].set_fontsize(10)
        table.scale(1, 1.8)
        plt.title("Final Model Performance Summary", fontsize=16, pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Subsequent Pages: Per-model before/after graphs
        base_model_names = [
            res["model_name"] for res in results_obj.before_mapie_results
        ]

        for name in base_model_names:
            res_before = next(
                (
                    r
                    for r in results_obj.before_mapie_results
                    if r["model_name"] == name
                ),
                None,
            )
            res_after = next(
                (
                    r
                    for r in results_obj.results
                    if r["model_name"] == f"{name} (With MAPIE)"
                ),
                None,
            )

            if not res_before:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            fig.suptitle(f"Before vs. After MAPIE Comparison: {name}", fontsize=18)

            # --- ROC Curve Plot ---
            fpr, tpr, _ = roc_curve(res_before["y_true"], res_before["y_prob"])
            ax1.plot(
                fpr,
                tpr,
                label=f"Before MAPIE (AUC = {res_before['metrics']['auc_score']:.3f})",
                lw=2,
            )
            if res_after:
                fpr_a, tpr_a, _ = roc_curve(res_after["y_true"], res_after["y_prob"])
                ax1.plot(
                    fpr_a,
                    tpr_a,
                    label=f"With MAPIE (AUC = {res_after['metrics']['auc_score']:.3f})",
                    lw=2,
                    linestyle="--",
                )
            ax1.plot([0, 1], [0, 1], "k--")
            ax1.set_title("ROC Curve")
            ax1.set_xlabel("False Positive Rate")
            ax1.set_ylabel("True Positive Rate")
            ax1.legend()
            ax1.grid(True)

            # --- Precision-Recall Curve Plot ---
            prec, rec, _ = precision_recall_curve(
                res_before["y_true"], res_before["y_prob"]
            )
            ax2.plot(rec, prec, label=f"Before MAPIE", lw=2)
            if res_after:
                prec_a, rec_a, _ = precision_recall_curve(
                    res_after["y_true"], res_after["y_prob"]
                )
                ax2.plot(rec_a, prec_a, label=f"With MAPIE", lw=2, linestyle="--")
            ax2.set_title("Precision-Recall Curve")
            ax2.set_xlabel("Recall")
            ax2.set_ylabel("Precision")
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close()

    print(f"✓ Detailed PDF report saved to {output_path}")

    # Log PDF as artifact to MLflow
    if mlflow_manager:
        with mlflow.start_run(run_id=mlflow_manager.parent_run_id, nested=True):
            mlflow.log_artifact(output_path, "reports")

    return {}


# --- Graph Definition & Execution ---
if __name__ == "__main__":
    # 1. Setup MLflow
    print("--- Setting up MLflow ---")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Use SQLite for local tracking
    mlflow_manager = MLflowManager("ML_Pipeline_LangGraph")
    mlflow_manager.start_parent_run("Pipeline_Execution")

    # 2. Load or generate sample data
    print("--- Loading/Generating Data ---")

    # Try to load your data here
    train_df = pd.read_csv("x_train.csv")
    test_df = pd.read_csv("x_test.csv")
    calib_df = pd.read_csv("x_calibrate.csv")
    target_col = "Churn"

    #     # For demo purposes, generate sample data
    #     np.random.seed(42)
    #     n_samples = 1000
    #     n_features = 10

    #     X = np.random.randn(n_samples, n_features)
    #     y = (X[:, 0] + X[:, 1] + 0.5 * np.random.randn(n_samples) > 0).astype(int)

    #     feature_names = [f"feature_{i}" for i in range(n_features)]
    #     df = pd.DataFrame(X, columns=feature_names)
    #     df["target"] = y

    #     print(f"Generated dataset with {n_samples} samples and {n_features} features")

    # except Exception as e:
    #     print(f"Error loading data: {e}")
    #     exit(1)

    # # 3. Split data
    # target_col = "target"
    # train_df, temp_df = train_test_split(
    #     df, test_size=0.4, random_state=42, stratify=df[target_col]
    # )
    # test_df, calib_df = train_test_split(
    #     temp_df, test_size=0.5, random_state=42, stratify=temp_df[target_col]
    # )

    # print(
    #     f"Data split - Train: {len(train_df)}, Test: {len(test_df)}, Calibration: {len(calib_df)}"
    # )

    # 4. Initialize state
    initial_state = MLPipelineState(
        x_train=train_df,
        x_test=test_df,
        x_calibrate=calib_df,
        target=target_col,
        model_results=ModelResults(mlflow_manager),
        best_model_info=None,
        decision="",
        mlflow_manager=mlflow_manager,
    )

    # 5. Build the graph
    print("--- Building LangGraph Workflow ---")
    workflow = StateGraph(MLPipelineState)

    # Add nodes
    workflow.add_node("initial_training", initial_model_training_node)
    workflow.add_node("evaluate_decide", evaluate_and_decide_node)
    workflow.add_node("tune_model", hyperparameter_tuning_node)
    workflow.add_node("calibrate_model", calibrate_model_node)
    workflow.add_node("final_report", final_report_node)

    # Set entry point
    workflow.set_entry_point("initial_training")

    # Add edges
    workflow.add_edge("initial_training", "evaluate_decide")

    # Add conditional edges
    workflow.add_conditional_edges(
        "evaluate_decide",
        lambda state: state["decision"],
        {
            "tune_best_model": "tune_model",
            "calibrate_best_model": "calibrate_model",
            "proceed_to_report": "final_report",
        },
    )

    workflow.add_edge("tune_model", "final_report")
    workflow.add_edge("calibrate_model", "final_report")
    workflow.add_edge("final_report", END)

    # Compile the graph
    app = workflow.compile()

    # 6. Execute the workflow
    print("--- Executing ML Pipeline ---")
    try:
        final_state = app.invoke(initial_state)

        # 7. Display final results
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 60)

        results_df = final_state["model_results"].get_results_df()
        if not results_df.empty:
            print("\nFinal Model Rankings:")
            print(results_df.to_string(index=False, float_format="%.4f"))

            best_model = results_df.iloc[0]["Model"]
            best_auc = results_df.iloc[0]["auc_score"]
            print(f"\nBest Model: {best_model} (AUC: {best_auc:.4f})")

        print(f"\nDecision taken: {final_state.get('decision', 'N/A')}")
        print("PDF report generated: model_comparison_report_v2.pdf")

        # 8. MLflow experiment info
        if mlflow_manager.parent_run_id:
            print(f"\nMLflow Run ID: {mlflow_manager.parent_run_id}")
            print("View results in MLflow UI with: mlflow ui")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # 9. Cleanup
        mlflow_manager.finish_run()
        print("\nPipeline completed successfully!")
