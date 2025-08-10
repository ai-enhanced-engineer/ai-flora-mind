# Research Directory

Machine learning experiments and analysis for iris classification.

[← Back to main README](../README.md)

## Quick Navigation

1. **Start here**: [EDA.ipynb](./eda/EDA.ipynb) - Key insights and analysis (5 cells)
2. **Full journey**: [EXPERIMENTS_JOURNEY.md](./EXPERIMENTS_JOURNEY.md) - Complete ML narrative
3. **Technical details**: Individual `EXPERIMENT.md` files in experiment directories

## Directory Structure

```
research/
├── EXPERIMENTS_JOURNEY.md       # Complete ML journey narrative
├── data.py                      # Data loading utilities
├── evaluation.py                # Model evaluation functions
├── features.py                  # Feature engineering module
├── eda/                         
│   └── EDA.ipynb                # Exploratory data analysis
├── experiments/                 
│   ├── base.py                  # Shared experiment functionality
│   ├── constants.py             # Centralized paths
│   ├── rule_based_heuristic/    # 96.0% baseline
│   ├── decision_tree/           # 96.7% best accuracy
│   ├── random_forest/           # 96.0% with regularization
│   └── xgboost/                 # 96.0% heavily tuned
├── models/                      # Trained models (.joblib)
└── results/                     # Experiment results (.json)
```

## Running Experiments

```bash
# Baseline
make eval-heuristic

# Decision Tree
make train-decision-tree              # Split validation
make train-decision-tree-comprehensive # LOOCV

# Random Forest  
make train-random-forest              # Split
make train-random-forest-comprehensive # Full validation
make train-random-forest-regularized  # Production-ready

# XGBoost
make train-xgboost                    # Conservative
make train-xgboost-comprehensive      # Full validation
make train-xgboost-optimized          # Heavy regularization
```

## Key Finding

All approaches converge at ~96% accuracy (Bayes error rate). See [EXPERIMENTS_JOURNEY.md](./EXPERIMENTS_JOURNEY.md) for complete analysis.

## Production Workflow

1. **Research**: Models trained in `research/models/` with timestamps
2. **Selection**: Best models chosen based on validation results
3. **Promotion**: Selected models copied to [`registry/prd/`](../registry/prd/)
4. **Deployment**: Docker image includes production registry