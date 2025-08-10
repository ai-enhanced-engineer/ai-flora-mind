# Random Forest Iris Classifier Experiment

## Executive Summary

Random Forest experiments revealed that proper validation and regularization are essential even for ensemble methods. Performance stabilized at 96% OOB accuracy, matching the simple heuristic baseline despite using 14 features versus 2.

**Key Results**:
- Split validation: 93.3% (underestimated due to data scarcity)
- Comprehensive: 96% OOB with severe overfitting (100% training)
- Regularized: 96% OOB with controlled overfitting (98.7% training)

## Experimental Design

### Objective
Evaluate Random Forest's ability to leverage complex feature interactions and validate EDA findings about petal feature dominance.

### Methodology
- **Features**: 14 total (4 original + 10 engineered)
- **Validation**: Split (70/30), OOB, and comprehensive LOOCV
- **Strategy**: Progressive regularization to control overfitting

---

## Experiment 1: Split Validation

### Results
- **Test Accuracy**: 93.3% (42/45 correct)
- **Training Accuracy**: 100%
- **Configuration**: 200 trees, default parameters

### Confusion Matrix
```
             Predicted
Actual    Setosa  Versicolor  Virginica
Setosa       15           0          0
Versicolor    0          14          1  
Virginica     0           2         13
```

**Finding**: Split validation underestimates performance due to limited test data.

---

## Experiment 2: Comprehensive Validation

### Results
- **OOB Score**: 96% (unbiased estimate)
- **LOOCV**: 96.0% ± 19.6%
- **10-Fold CV**: 96.1% ± 4.5%
- **Training Accuracy**: 100% (perfect memorization)

### Key Discovery
**Severe overfitting**: 4% gap between training (100%) and OOB (96%) indicates memorization despite ensemble nature.

---

## Experiment 3: Regularized Configuration

### Optimization
```python
RandomForestClassifier(
    n_estimators=100,        # Reduced from 300
    max_depth=5,             # Limited depth  
    min_samples_split=5,     # Increased from 2
    min_samples_leaf=2,      # Increased from 1
    max_features='sqrt'      # Feature subsampling
)
```

### Results
- **OOB Score**: 96.67%
- **Training Accuracy**: 98.67%
- **Overfitting Gap**: 2.0% (reduced by 50%)

---

## Comparative Analysis

| Metric | Split | Comprehensive | Regularized | Change |
|--------|-------|---------------|-------------|---------|
| Test/OOB Accuracy | 93.3% | 96.0% | 96.0% | +2.7% |
| Training Accuracy | 100% | 100% | 98.7% | -1.3% |
| Overfitting Gap | 6.7% | 4.0% | 2.7% | -60% |
| Trees | 200 | 300 | 100 | -67% |

### Feature Importance Stability
- Combined petal importance: 65-70%
- area_ratio most stable: 15.2% → 16.6%
- Engineered features: 60-65% total importance

---

## Conclusions

1. **Performance ceiling**: Random Forest achieves 96% accuracy, matching heuristic baseline despite 7x more features.

2. **Overfitting universal**: Even ensemble methods overfit on small datasets without regularization.

3. **OOB validation critical**: Revealed true 96% performance versus misleading 100% training accuracy.

4. **Optimal configuration**: 100 trees with depth/sample constraints provides best accuracy-complexity trade-off.

5. **Feature engineering validated**: Engineered features (area_ratio, petal_area) consistently rank top.