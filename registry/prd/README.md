# Production Models Registry

This directory contains the production-ready models selected from our research experiments. Each model was chosen based on rigorous validation and specific criteria for production deployment.

## Model Selection Criteria

### decision_tree.joblib
- **Source**: `decision_tree_comprehensive_2025_07_22_000131.joblib`
- **Accuracy**: 96.7% (LOOCV validation)
- **Why chosen**: Highest accuracy among all models. The comprehensive LOOCV validation ensures robust performance across all data points, significantly outperforming the 70/30 split variant (91.1%).

### random_forest.joblib
- **Source**: `random_forest_regularized_2025_07_22_000255.joblib`
- **Accuracy**: 96.0% (Full dataset evaluation)
- **Why chosen**: Production-optimized with regularization constraints (max_depth=5, min_samples_split=5) to prevent overfitting while maintaining the same 96% accuracy as the comprehensive model. Better suited for production than the unregularized version.

### xgboost.joblib
- **Source**: `xgboost_optimized_2025_07_22_000453.joblib`
- **Accuracy**: 96.0% (LOOCV validation)
- **Why chosen**: Best performing XGBoost variant with heavy regularization. Outperforms both the comprehensive (94.7%) and split (93.3%) models through careful hyperparameter tuning focused on generalization.

## Key Insight
All production models converge around 96% accuracy, confirming this as the practical performance ceiling for the Iris dataset. The decision tree slightly edges ahead at 96.7%, demonstrating that simpler models can match or exceed complex ensemble methods on this well-structured dataset.