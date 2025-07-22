"""
Random Forest Iris Classifier - Regularized Experiment

This module implements the regularized experiment for Random Forest on the Iris dataset.
Heavy regularization to prevent overfitting on the small dataset.

Configuration:
- Reduced trees: 100 instead of 300
- Depth limit: max_depth=5
- Sample constraints: min_samples_split=5, min_samples_leaf=2
- Feature subsampling: max_features='sqrt'
"""

import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ai_flora_mind.logging import get_logger
from research.data import load_iris_data
from research.evaluation import evaluate_model, log_performance_summary
from research.experiments.constants import MODELS_DIR, RESULTS_DIR
from research.features import ModelType, engineer_features

logger = get_logger(__name__)


def main() -> None:
    """Run Random Forest regularized experiment with overfitting prevention."""
    logger.info("Starting Random Forest Regularized Experiment")

    # Load Iris dataset
    logger.info("Loading Iris dataset")
    X_original, y, iris_data = load_iris_data()
    feature_names_original = iris_data.feature_names
    logger.info(
        "Dataset loaded successfully",
        features=feature_names_original,
        target_classes=list(set(y)),
        total_samples=len(X_original),
    )

    # Log petal feature statistics for context
    petal_length_idx = feature_names_original.index("petal length (cm)")
    petal_width_idx = feature_names_original.index("petal width (cm)")
    petal_length_range = f"{X_original[:, petal_length_idx].min():.2f}-{X_original[:, petal_length_idx].max():.2f}"
    petal_width_range = f"{X_original[:, petal_width_idx].min():.2f}-{X_original[:, petal_width_idx].max():.2f}"
    logger.info("Petal feature statistics", petal_length_range=petal_length_range, petal_width_range=petal_width_range)

    # Engineer features for Random Forest (all 14 features)
    logger.info(
        "Engineering features for Random Forest",
        n_samples=len(X_original),
        original_features=len(feature_names_original),
    )
    X_engineered, all_feature_names = engineer_features(X_original, feature_names_original, ModelType.RANDOM_FOREST)
    logger.info("Feature engineering completed", total_features=X_engineered.shape[1])

    logger.info("Running regularized experiment (overfitting prevention)")

    # Step 1: Train regularized model on full dataset
    logger.info("Step 1: Training regularized model on full dataset", total_samples=len(X_engineered))
    model = RandomForestClassifier(
        n_estimators=100,  # Reduced from 300
        max_depth=5,  # Added depth limit
        min_samples_split=5,  # Increased from 2
        min_samples_leaf=2,  # Increased from 1
        max_features="sqrt",  # Feature subsampling
        random_state=42,
        n_jobs=-1,
        oob_score=True,  # Enable out-of-bag scoring for validation
    )

    logger.info(
        "Training regularized Random Forest classifier",
        features=len(all_feature_names),
        n_estimators=100,
        max_depth=5,
        samples=len(X_engineered),
    )

    model.fit(X_engineered, y)

    # Log training completion with OOB score
    training_accuracy = model.score(X_engineered, y)
    oob_score = model.oob_score_
    overfitting_gap = training_accuracy - oob_score
    logger.info(
        "Regularized Random Forest training completed",
        training_accuracy=training_accuracy,
        oob_score=oob_score,
        overfitting_gap=overfitting_gap,
        n_estimators=model.n_estimators,
    )

    # Extract feature importance for validation
    importance_scores = model.feature_importances_
    feature_importance = dict(zip(all_feature_names, importance_scores))
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    top_features = list(sorted_importance.keys())[:5]
    top_feature = top_features[0]
    top_importance = list(sorted_importance.values())[0]
    logger.info("Top 5 most important features", features=top_features)
    logger.info("Feature importance extracted", top_feature=top_feature, top_importance=top_importance)

    # Validate EDA findings: petal features should dominate
    petal_features = [f for f in top_features if "petal" in f.lower()]
    if len(petal_features) >= 2:
        logger.info("EDA validation successful", insight="Petal features dominate as expected")
    else:
        logger.warning(
            "EDA validation concern", insight="Petal features not dominating as expected", top_features=top_features
        )

    # Generate OOB predictions for proper evaluation
    # OOB predictions are more reliable as they're based on trees that didn't see each sample
    oob_decision = model.oob_decision_function_
    y_pred_oob = np.argmax(oob_decision, axis=1)

    # Use evaluate_model with OOB predictions for proper metrics
    eval_results = evaluate_model(y, y_pred_oob, iris_data, X_engineered)

    # Build regularized results
    results = {
        "predictions": model.predict(X_engineered).tolist(),
        "true_labels": y.tolist(),
        "algorithm_details": {
            "experiment_type": "regularized",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "features_used": all_feature_names,
            "feature_importance": sorted_importance,
            "training_samples": len(X_engineered),
            "training_accuracy": float(training_accuracy),
            "oob_score": float(oob_score),
            "overfitting_gap": float(overfitting_gap),
        },
        "overfitting_analysis": {
            "training_accuracy": float(training_accuracy),
            "oob_score": float(oob_score),
            "performance_gap": float(overfitting_gap),
            "gap_assessment": "Reduced overfitting" if overfitting_gap < 0.035 else "Still overfitting",
            "regularization_applied": {
                "reduced_trees": "300 → 100",
                "added_depth_limit": "None → 5",
                "increased_sample_constraints": "2,1 → 5,2",
                "feature_subsampling": "all → sqrt",
            },
        },
        "full_dataset_training": {
            "training_accuracy": float(training_accuracy),
            "oob_score": float(oob_score),
            "total_samples": len(X_engineered),
            "samples_per_class": len(X_engineered) // 3,  # Assuming equal distribution
        },
        # Add classification results
        "per_class_accuracy": eval_results["per_class_accuracy"],
        "confusion_matrix": eval_results["confusion_matrix"],
        "classification_report": eval_results["classification_report"],
        "misclassifications": eval_results["misclassifications"],
    }

    # Add performance metrics section
    results["performance_metrics"] = {
        "overall_accuracy": eval_results["overall_accuracy"],
        "total_samples": eval_results["total_samples"],
        "correct_predictions": eval_results["correct_predictions"],
        "misclassification_count": len(eval_results["misclassifications"]),
    }

    # Performance validation against overfitting
    if overfitting_gap <= 0.03:
        logger.info(
            "Overfitting successfully reduced",
            achieved_gap=overfitting_gap,
            target_gap="< 0.03",
            insight=f"Gap reduced to {overfitting_gap:.3f}",
        )
    elif overfitting_gap <= 0.035:
        logger.warning(
            "Overfitting partially reduced",
            achieved_gap=overfitting_gap,
            target_gap="< 0.03",
            warning=f"Gap {overfitting_gap:.3f} still above ideal",
        )
    else:
        logger.error(
            "Overfitting not resolved",
            achieved_gap=overfitting_gap,
            target_gap="< 0.03",
            error=f"Gap {overfitting_gap:.3f} still too high",
        )

    # Log performance summary
    experiment_name = "random_forest_regularized"
    logger.info("Step 4: Evaluating model performance")
    log_performance_summary(results, experiment_name)

    # Save model and results
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    model_filename = f"{experiment_name}_{timestamp}.joblib"
    results_filename = f"{experiment_name}_{timestamp}.json"

    # Use centralized directory constants
    results_dir = RESULTS_DIR
    models_dir = MODELS_DIR

    # Save model
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)

    # Log model details
    model_size = os.path.getsize(model_path)
    logger.info("Model saved", path=model_path, size_bytes=model_size, n_estimators=model.n_estimators)

    # Save results
    results_path = os.path.join(results_dir, results_filename)
    import json

    # Convert any numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj

    json_safe_results = convert_numpy_to_list(results)

    with open(results_path, "w") as f:
        json.dump(json_safe_results, f, indent=2)
    logger.info("Results saved", results_path=results_path)

    # Final summary
    summary = (
        f"Random Forest Regularized Experiment: {results['performance_metrics']['overall_accuracy']:.1%} OOB accuracy. "
        f"Training accuracy: {training_accuracy:.1%}. "
        f"Overfitting gap: {overfitting_gap:.1%}. "
        f"Top features: {', '.join(top_features[:3])}"
    )

    logger.info(
        "Random Forest regularized experiment completed successfully",
        accuracy=f"{results['performance_metrics']['overall_accuracy']:.1%}",
        model_saved=model_filename,
        results_saved=results_filename,
    )

    logger.info(summary)


if __name__ == "__main__":
    main()
