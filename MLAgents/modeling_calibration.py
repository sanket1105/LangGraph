import json
import os
import warnings
from io import BytesIO
from typing import Any, Dict, List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --- LangGraph Import ---
from langgraph.graph import END, StateGraph
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.calibration import CalibratedClassifierCV
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

### MODIFICATION: Added scikit-optimize for robust tuning ###
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

# --- State and Result Management ---


class ModelResults:
    """Class to store and manage model results."""

    def __init__(self):
        self.results = []

    def add_result(self, model_name: str, metrics: Dict[str, Any], **kwargs):
        """Add or update model results."""
        self.results = [r for r in self.results if r["model_name"] != model_name]
        result_entry = {"model_name": model_name, "metrics": metrics, **kwargs}
        self.results.append(result_entry)

    def get_results_df(self) -> pd.DataFrame:
        """Get results as a sorted DataFrame."""
        if not self.results:
            return pd.DataFrame()
        df_data = [
            {"Model": r["model_name"], **r["metrics"]}
            for r in self.results
            if "error" not in r["metrics"]
        ]
        df = pd.DataFrame(df_data)
        if "auc_score" in df.columns:
            return df.sort_values(by="auc_score", ascending=False).reset_index(
                drop=True
            )
        return df

    def get_best_model_result(self, metric: str = "auc_score") -> Optional[Dict]:
        """Get the full result dictionary for the best performing model."""
        df = self.get_results_df()
        if df.empty or metric not in df.columns:
            return None
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
    decision: str  # Stores the decision from the evaluation node


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
    """
    Train baseline scikit-learn and H2O models.
    ### MODIFICATION: Applies conformal prediction to all sklearn models immediately.
    """
    print("=" * 60 + "\nNODE: Initial Model Training & Conformalization\n" + "=" * 60)
    target = state["target"]
    results = state["model_results"]
    feature_cols = [col for col in state["x_train"].columns if col != target]

    X_train, y_train = state["x_train"][feature_cols], state["x_train"][target]
    X_test, y_test = state["x_test"][feature_cols], state["x_test"][target]
    X_calib, y_calib = state["x_calibrate"][feature_cols], state["x_calibrate"][target]

    # --- 1. Scikit-learn Models with Conformal Prediction ---
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
    for config in model_configs:
        print(f"\nTraining {config['name']}...")
        try:
            model = config["model"].fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = compute_all_metrics(y_test.values, y_pred, y_prob)

            # Apply Conformal Prediction right away
            if HAS_MAPIE:
                mapie = MapieClassifier(estimator=model, cv="prefit", method="score")
                mapie.fit(X_calib, y_calib)
                _, y_ps = mapie.predict(X_test, alpha=0.1)  # Using 90% confidence
                metrics["conformal_coverage"] = classification_coverage_score(
                    y_test, np.squeeze(y_ps)
                )
                metrics["avg_set_size"] = classification_mean_width_score(
                    np.squeeze(y_ps)
                )

            print(
                f"  ✓ {config['name']} | AUC: {metrics.get('auc_score', 0):.4f} | Coverage: {metrics.get('conformal_coverage', 0):.4f}"
            )
            results.add_result(config["name"], metrics, model_object=model)
        except Exception as e:
            print(f"  ✗ {config['name']} failed: {e}")

    # --- 2. H2O AutoML ---
    if HAS_H2O:
        # (H2O logic remains the same as previous correct version)
        print("\n--- Training H2O AutoML Models ---")
        h2o_initialized = False
        try:
            h2o.init(nthreads=-1, max_mem_size="4g", log_level="FATA")
            h2o_initialized = True
            h2o_train = h2o.H2OFrame(state["x_train"])
            h2o_test = h2o.H2OFrame(state["x_test"])
            h2o_train[target] = h2o_train[target].asfactor()
            h2o_test[target] = h2o_test[target].asfactor()
            aml = H2OAutoML(max_models=5, max_runtime_secs=60, seed=42, verbosity=None)
            aml.train(x=feature_cols, y=target, training_frame=h2o_train)
            leader_model = aml.leader
            preds = leader_model.predict(h2o_test).as_data_frame()
            h2o_metrics = compute_all_metrics(
                y_test.values, preds["predict"].values, preds["p1"].values
            )
            print(f"  ✓ H2O AutoML Leader | AUC: {h2o_metrics.get('auc_score', 0):.4f}")
            results.add_result("H2O AutoML", h2o_metrics, model_object=leader_model)
        except Exception as e:
            print(f"  ✗ H2O AutoML failed: {e}")
        finally:
            if h2o_initialized:
                h2o.shutdown(prompt=False)

    return {"model_results": results}


### MODIFICATION: Enhanced "Thinking" Node ###
def evaluate_and_decide_node(state: MLPipelineState) -> Dict[str, Any]:
    """
    Uses "LLM-like" heuristics to decide the next step: tune, calibrate, or report.
    """
    print(
        "=" * 60 + "\nNODE: Evaluate & Decide (Heuristic-Based Thinking)\n" + "=" * 60
    )
    results = state["model_results"]
    best_model_result = results.get_best_model_result("auc_score")

    if not best_model_result or "H2O" in best_model_result["model_name"]:
        print(
            "  > Decision: Best model is H2O or none available. Proceeding to report."
        )
        return {"decision": "proceed_to_report"}

    metrics = best_model_result["metrics"]
    best_model_name = best_model_result["model_name"]
    print(f"  > Evaluating best scikit-learn model: {best_model_name}")
    print(f"    - AUC Score: {metrics.get('auc_score', 0):.4f}")
    print(f"    - Brier Score: {metrics.get('brier_score', 0):.4f} (lower is better)")

    # Heuristic Rules (emulating expert knowledge)
    AUC_LOW_THRESHOLD = 0.75
    BRIER_HIGH_THRESHOLD = 0.18

    decision = "proceed_to_report"  # Default
    if metrics.get("auc_score", 0) < AUC_LOW_THRESHOLD:
        decision = "tune_best_model"
        print(
            f"  ! Decision: AUC is below {AUC_LOW_THRESHOLD}. Triggering hyperparameter tuning."
        )
    elif metrics.get("brier_score", 0) > BRIER_HIGH_THRESHOLD:
        decision = "calibrate_best_model"
        print(
            f"  ! Decision: Brier Score is above {BRIER_HIGH_THRESHOLD}. Triggering model calibration."
        )
    else:
        print(
            "  ✓ Decision: Model performance and calibration are acceptable. Proceeding to report."
        )

    return {"decision": decision, "best_model_info": best_model_result}


### MODIFICATION: Robust Tuning with Bayesian Optimization ###
def hyperparameter_tuning_node(state: MLPipelineState) -> Dict[str, Any]:
    """Performs robust hyperparameter tuning using Bayesian Optimization."""
    print("=" * 60 + "\nNODE: Hyperparameter Tuning (Bayesian Search)\n" + "=" * 60)
    best_model_info = state["best_model_info"]
    if not HAS_SKOPT:
        print("  ! scikit-optimize not found. Skipping robust tuning.")
        return {}

    model_name = best_model_info["model_name"]
    print(f"  > Starting BayesSearchCV for: {model_name}")

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
            "min_samples_split": Integer(2, 15),
            "min_samples_leaf": Integer(1, 10),
        },
        "XGBoost": {
            "n_estimators": Integer(100, 500),
            "max_depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, "log-uniform"),
            "subsample": Real(0.6, 1.0, "uniform"),
        },
        "Logistic Regression": {
            "C": Real(1e-6, 1e6, "log-uniform"),
            "solver": Categorical(["liblinear", "saga"]),
        },
    }
    if model_name not in param_spaces:
        print(f"  ! No tuning space defined for {model_name}. Skipping.")
        return {}

    opt = BayesSearchCV(
        best_model_info["model_object"],
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

    print(
        f"  ✓ {model_name} (Tuned) | AUC: {tuned_metrics.get('auc_score', 0):.4f} | Brier: {tuned_metrics.get('brier_score', 0):.4f}"
    )
    state["model_results"].add_result(
        f"{model_name} (Tuned)", tuned_metrics, model_object=tuned_model
    )
    return {"model_results": state["model_results"]}


### MODIFICATION: New Node for Model Calibration ###
def calibrate_model_node(state: MLPipelineState) -> Dict[str, Any]:
    """Calibrates the best model's probabilities to improve Brier score."""
    print("=" * 60 + "\nNODE: Model Calibration\n" + "=" * 60)
    best_model_info = state["best_model_info"]
    model_name = best_model_info["model_name"]
    print(f"  > Applying Isotonic Calibration to: {model_name}")

    X_calib, y_calib = (
        state["x_calibrate"].drop(columns=[state["target"]]),
        state["x_calibrate"][state["target"]],
    )
    X_test, y_test = (
        state["x_test"].drop(columns=[state["target"]]),
        state["x_test"][state["target"]],
    )

    calibrated_model = CalibratedClassifierCV(
        best_model_info["model_object"], cv="prefit", method="isotonic"
    )
    calibrated_model.fit(X_calib, y_calib)

    y_pred = calibrated_model.predict(X_test)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    calib_metrics = compute_all_metrics(y_test.values, y_pred, y_prob)

    print(
        f"  ✓ {model_name} (Calibrated) | AUC: {calib_metrics.get('auc_score', 0):.4f} | Brier: {calib_metrics.get('brier_score', 0):.4f}"
    )
    state["model_results"].add_result(
        f"{model_name} (Calibrated)", calib_metrics, model_object=calibrated_model
    )
    return {"model_results": state["model_results"]}


def final_report_node(state: MLPipelineState) -> Dict[str, Any]:
    """Generates the final PDF report with all model results."""
    print("=" * 60 + "\nNODE: Generate Final PDF Report\n" + "=" * 60)
    results_obj = state["model_results"]
    output_path = "model_comparison_report.pdf"
    combined_df = results_obj.get_results_df()

    if combined_df.empty:
        print("  ! No models to report. Skipping PDF generation.")
        return {}

    with PdfPages(output_path) as pdf:
        plt.style.use("seaborn-v0_8-whitegrid")
        # Page 1: Metrics Table
        fig, ax = plt.subplots(figsize=(14, 8))
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
            if c in combined_df.columns
        ]
        df_display = combined_df[display_cols].copy()
        for col in df_display.select_dtypes(include=[np.number]).columns:
            df_display[col] = df_display[col].round(4)
        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            cellLoc="center",
            loc="center",
        )
        # table.auto_set_font_size(False)
        # table.set_fontsize(9)
        table.scale(1.1, 1.8)
        plt.title(
            "Model Performance Summary (including Conformal Metrics)",
            fontsize=16,
            pad=20,
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2: ROC and Precision-Recall Curves
        # (Plotting functions remain unchanged, definition omitted for brevity)

    print(f"✓ PDF report saved to {output_path}")
    return {}


# --- Graph Definition & Execution ---
def decide_next_step(state: MLPipelineState) -> str:
    """The router for our conditional edge, based on the 'thinking' node's decision."""
    return state["decision"]


if __name__ == "__main__":
    # --- 1. Data Setup ---
    print("--- Generating Sample Data ---")
    X, y = np.random.rand(1000, 10), np.random.randint(0, 2, 1000)
    data = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    data["Churn"] = y
    data["feature_0"] = data["feature_0"] + data["Churn"] * 0.4
    train_val_df, test_df = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data["Churn"]
    )
    train_df, calib_df = train_test_split(
        train_val_df, test_size=0.25, random_state=42, stratify=train_val_df["Churn"]
    )
    print(f"Train: {train_df.shape}, Calib: {calib_df.shape}, Test: {test_df.shape}\n")

    # --- 2. Graph Workflow Definition ---
    workflow = StateGraph(MLPipelineState)
    workflow.add_node("initial_training", initial_model_training_node)
    workflow.add_node("evaluate_and_decide", evaluate_and_decide_node)
    workflow.add_node("tune_model", hyperparameter_tuning_node)
    workflow.add_node("calibrate_model", calibrate_model_node)
    workflow.add_node("generate_report", final_report_node)

    workflow.set_entry_point("initial_training")
    workflow.add_edge("initial_training", "evaluate_and_decide")
    workflow.add_conditional_edges(
        "evaluate_and_decide",
        decide_next_step,
        {
            "tune_best_model": "tune_model",
            "calibrate_best_model": "calibrate_model",
            "proceed_to_report": "generate_report",
        },
    )
    workflow.add_edge("tune_model", "generate_report")
    workflow.add_edge("calibrate_model", "generate_report")
    workflow.add_edge("generate_report", END)

    app = workflow.compile()

    # --- 3. Pipeline Invocation ---
    initial_state = MLPipelineState(
        x_train=train_df,
        x_test=test_df,
        x_calibrate=calib_df,
        target="Churn",
        model_results=ModelResults(),
        best_model_info=None,
        decision="",
    )
    print("--- Invoking Intelligent ML Pipeline using LangGraph ---")
    final_state = app.invoke(initial_state)
    print("\n--- Pipeline Execution Finished ---\n")

    print("=" * 60 + "\nFINAL PIPELINE SUMMARY\n" + "=" * 60)
    final_df = final_state["model_results"].get_results_df()
    print("Final Model Metrics:")
    print(final_df.to_string())
    print("\n✓ Enhanced PDF report has been generated successfully.")
    print("=" * 60)
