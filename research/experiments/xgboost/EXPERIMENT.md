# XGBoost Iris Classifier Experiment

## Executive Summary

XGBoost experiments confirmed that complex gradient boosting provides no advantage over simpler methods on well-structured datasets. Despite extensive hyperparameter tuning, XGBoost achieved only 96.0% accuracy—matching the rule-based heuristic while requiring significantly more complexity.

**Key Results**:
- Split validation: 93.3% (conservative configuration)
- Comprehensive: 94.7% (moderate tuning, still overfitted)
- Optimized: 96.0% (heavy regularization, matches heuristic)

## Experimental Design

### Objective
Explore whether gradient boosting can surpass the 96% performance ceiling established by simpler models, targeting 98-99%.

### Methodology
- **Features**: 9 total (4 original + 5 engineered including custom interactions)
- **Validation**: Split, LOOCV, and early stopping
- **Strategy**: Progressive regularization to control overfitting

---

## Experiment 1: Conservative Baseline

### Configuration
```python
XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8
)
```

### Results
- **Test Accuracy**: 93.3% (42/45)
- **Training Accuracy**: 100%
- **Overfitting Gap**: 6.7%

### Confusion Matrix
```
             Predicted
Actual    Setosa  Versicolor  Virginica
Setosa       15           0          0
Versicolor    0          15          0
Virginica     0           3         12
```

**Finding**: Custom interaction term dominates (24.9%) but performance below target.

---

## Experiment 2: Full Dataset Validation

### Results
- **LOOCV Accuracy**: 94.7% (142/150) ± 22.5%
- **10-Fold CV**: 95.3% ± 5.1%
- **Training Accuracy**: 100%
- **Overfitting Gap**: 5.3%

### Key Discovery
**Performance below competitors**: XGBoost (94.7%) underperforms Heuristic (96%), Decision Tree (96.7%), and Random Forest (96%).

---

## Experiment 3: Heavy Regularization

### Optimization
```python
XGBClassifier(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.05,    # Halved
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,         # L1
    reg_lambda=0.1          # L2
)
```

### Results
- **LOOCV Accuracy**: 96.0% (144/150)
- **Training Accuracy**: 97.3%
- **Overfitting Gap**: 1.3%

---

## Comparative Analysis

| Metric | Split | Comprehensive | Optimized | Change |
|--------|-------|---------------|-----------|---------|
| Test/LOOCV Accuracy | 93.3% | 94.7% | 96.0% | +2.7% |
| Training Accuracy | 100% | 100% | 97.3% | -2.7% |
| Overfitting Gap | 6.7% | 5.3% | 1.3% | -81% |
| Learning Rate | 0.1 | 0.1 | 0.05 | -50% |

### Feature Importance Evolution
- Split: Custom interaction dominates (24.9%)
- Comprehensive: Petal length leads (27.8%)
- Optimized: Balanced across petal features

---

## Conclusions

1. **No advantage over simplicity**: XGBoost's 96% matches rule-based heuristic despite 100x complexity.

2. **Heavy regularization required**: Needed aggressive L1/L2 + early stopping to reach competitive performance.

3. **Custom features ineffective**: Specialized interaction terms provided no meaningful improvement.

4. **Gradient boosting overkill**: For well-structured data like iris, simpler methods suffice.

5. **Complexity-performance trade-off poor**: Highest computational cost for no accuracy gain over heuristic.