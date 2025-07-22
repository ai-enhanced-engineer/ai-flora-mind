# Research Directory

This directory contains all research experiments, analysis, and findings for the AI Flora Mind iris classification project.

## Understanding the Research

1. **Start here**: Read [EXPERIMENTS_JOURNEY.md](./EXPERIMENTS_JOURNEY.md) for the complete narrative
2. **Technical details**: Check individual `EXPERIMENT.md` files in each experiment directory
3. **Raw results**: JSON files in `results/` contain detailed metrics and predictions
4. **Trained models**: Serialized models in `models/` can be loaded with joblib

## Directory Structure

```
research/
├── README.md                    # This file
├── EXPERIMENTS_JOURNEY.md       # Comprehensive narrative of our ML journey
├── data.py                      # Data loading utilities
├── evaluation.py                # Model evaluation functions
├── features.py                  # Feature engineering module
├── eda/                         # Exploratory Data Analysis
│   ├── EDA.ipynb                # Jupyter notebook with analysis
│   └── MODELING_STRATEGY.md     # Modeling approach documentation
├── experiments/                 # All ML experiments
│   ├── constants.py             # Centralized directory paths
│   ├── rule_based_heuristic/    # Simple threshold-based baseline
│   │   ├── EXPERIMENT.md
│   │   └── iris_heuristic_classifier.py
│   ├── decision_tree/           # Decision tree experiments
│   │   ├── EXPERIMENT.md
│   │   ├── split.py             # 70/30 split validation
│   │   └── comprehensive.py     # LOOCV validation
│   ├── random_forest/           # Random forest experiments
│   │   ├── EXPERIMENT.md
│   │   ├── split.py             # 70/30 split
│   │   ├── comprehensive.py     # Full validation
│   │   └── regularized.py       # Production-optimized
│   └── xgboost/                 # XGBoost experiments
│       ├── EXPERIMENT.md
│       ├── split.py             # Conservative baseline
│       ├── comprehensive.py     # Full validation
│       └── optimized.py         # Heavy regularization
├── models/                      # Saved trained models (.joblib files)
└── results/                     # Experiment results (.json files)
```

## Key Findings Summary

| Model | Best Accuracy | Configuration | Validation Method |
|-------|---------------|---------------|-------------------|
| Rule-based Heuristic | 96.0% | 2 rules, 2 features | Full dataset |
| Decision Tree | 96.7% | max_depth=3 | LOOCV |
| Random Forest | 96.0% | 100 trees, regularized | OOB + LOOCV |
| XGBoost | 96.0% | 150 trees, heavy reg | LOOCV |

**Key Insight**: All approaches converge at ~96% accuracy, suggesting this is the Bayes error rate for the Iris dataset.

## Experiment Result Structure

All comprehensive experiments generate JSON files with:
- Full prediction arrays for all samples
- Individual validation scores (LOOCV, cross-validation folds)
- Confusion matrices and classification reports
- Feature importance rankings
- Misclassification analysis with sample details

## Dependencies

The research code uses:
- scikit-learn (decision trees, random forest, evaluation metrics)
- xgboost (gradient boosting)
- numpy, pandas (data manipulation)
- matplotlib, seaborn (EDA visualizations)

## Running Experiments

All experiments can be run using make commands:

```bash
# Rule-based heuristic (baseline)
make eval-heuristic

# Decision Tree experiments
make train-decision-tree              # 70/30 split validation
make train-decision-tree-comprehensive # LOOCV + repeated k-fold validation

# Random Forest experiments  
make train-random-forest              # 70/30 split
make train-random-forest-comprehensive # Full validation (OOB + LOOCV)
make train-random-forest-regularized  # Production-optimized configuration

# XGBoost experiments
make train-xgboost                    # Conservative baseline
make train-xgboost-comprehensive      # Full validation with LOOCV
make train-xgboost-optimized          # Heavy regularization
```

All experiment results are saved to:
- `models/`: Trained model files (.joblib)
- `results/`: Detailed metrics and predictions (.json)

## For More Information

See [EXPERIMENTS_JOURNEY.md](./EXPERIMENTS_JOURNEY.md) for the complete story of our machine learning journey with the Iris dataset.