# XGBoost Iris Classifier Experiment

## Executive Summary

XGBoost experiments confirmed that complex gradient boosting provides no advantage over simpler methods on well-structured datasets. Despite extensive hyperparameter tuning and custom feature engineering, XGBoost achieved only 96.0% accuracy—matching the rule-based heuristic while requiring significantly more complexity.

**Key Results**:
- Split validation: 93.3% (conservative configuration)
- Comprehensive: 94.7% (moderate tuning, still overfitted)
- Optimized: 96.0% (heavy regularization, matches heuristic)

## Experimental Design

### Objective
Explore whether gradient boosting can surpass the 96% performance ceiling established by simpler models, targeting theoretical maximum of 98-99%.

### Feature Engineering
- **4 original**: Standard iris measurements
- **5 engineered**: Including custom `versicolor_virginica_interaction`
- **Total**: 9 targeted features (vs Random Forest's 14)

### Validation Strategy
Three simultaneous experiments exploring different configurations due to XGBoost's complex hyperparameter space.

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

### Top Features
1. versicolor_virginica_interaction: 24.9%
2. petal width: 23.1%
3. petal length: 19.8%

**Finding**: Custom interaction term dominates but performance still below target.

---

## Experiment 2: Full Dataset Validation

### Configuration Changes
- n_estimators: 200 (doubled)
- max_depth: 4 (increased)
- Validation: LOOCV on 150 samples

### Results
- **LOOCV Accuracy**: 94.7% (142/150) ± 22.5%
- **Repeated 10-Fold CV**: 95.3% ± 5.1% (100 iterations)
- **Training Accuracy**: 100%
- **Overfitting Gap**: 5.3%

### Key Discovery
**Performance below all competitors**: Even with full dataset, XGBoost (94.7%) underperforms Heuristic (96%), Decision Tree (96.7%), and Random Forest (96%).

### Feature Importance Shift
1. petal length: 27.8% (takes lead)
2. petal_area: 23.6%
3. petal width: 22.8%
4. versicolor_virginica_interaction: 3.4% (drops dramatically)

---

## Experiment 3: Heavy Regularization

### Optimization Applied
```python
XGBClassifier(
    n_estimators=150,
    max_depth=3,           # Reduced back
    learning_rate=0.05,    # Halved
    subsample=0.7,         # More aggressive
    colsample_bytree=0.7,
    min_child_weight=3,    # Added
    gamma=0.1,             # Added
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=0.1,        # L2 regularization
    early_stopping_rounds=20
)
```

### Results
- **LOOCV Accuracy**: 96.0% (144/150) ± 19.6%
- **Training Accuracy**: 98.0%
- **Overfitting Gap**: 2.0% (70% reduction)

### Final Confusion Matrix
```
             Predicted
Actual    Setosa  Versicolor  Virginica
Setosa       50           0          0
Versicolor    1          46          3
Virginica     0           2         48
```

### Misclassifications
- Samples 70, 77, 83 (Versicolor → Virginica)
- Sample 98 (Versicolor → Setosa, unique error)
- Samples 119, 133 (Virginica → Versicolor)

---

## Comparative Analysis

### Performance Evolution
| Metric | Split | Comprehensive | Optimized | Trend |
|--------|-------|---------------|-----------|-------|
| Accuracy | 93.3% | 94.7% | 96.0% | +2.7% |
| Training | 100% | 100% | 98.0% | Controlled |
| Gap | 6.7% | 5.3% | 2.0% | -70% |
| Parameters | 5 | 5 | 11 | Complex |

### Feature Importance Dynamics
- Split: Custom interaction dominates (24.9%)
- Comprehensive: Original features rise (74.2% combined)
- Optimized: Balanced importance, interaction rebounds (13.6%)

---

## Conclusions

1. **No advantage over baseline**: XGBoost's 96.0% matches simple heuristic despite 100x complexity.

2. **Overfitting sensitivity**: Required aggressive regularization (11 hyperparameters) to control.

3. **Custom features limited value**: versicolor_virginica_interaction provided only marginal benefit.

4. **Convergence confirmed**: All models (Heuristic, DT, RF, XGB) converge at ~96% accuracy.

5. **Complexity not justified**: Simple 2-rule heuristic achieves same performance with interpretability.