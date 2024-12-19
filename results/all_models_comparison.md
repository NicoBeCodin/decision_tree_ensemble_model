# Comparaison des Modèles

## Résultats des Performances

| Modèle | MSE | Temps d'entraînement (s) | Temps d'évaluation (s) |
|--------|-----|------------------------|----------------------|
| Bagging | 1.460e-04 | 5.296 | 0.018 |

### Paramètres de Bagging

- max_depth: 60.000
- min_impurity_decrease: 0.000
- min_samples_split: 2.000
- n_estimators: 20.000

### Importance des Caractéristiques pour Bagging

| Caractéristique | Importance (%) |
|----------------|----------------|
| matrix_size_x | 1.43 |
| matrix_size_y | 1.77 |
| p1 | 10.40 |
| p2 | 1.69 |
| p3 | 46.82 |
| p4 | 16.83 |
| p5 | 5.13 |
| p6 | 21.24 |
| p7 | -8.88 |
| p8 | 3.58 |

---

| Boosting | 1.795e-04 | 6.724 | 0.016 |

### Paramètres de Boosting

- learning_rate: 0.070
- max_depth: 15.000
- min_impurity_decrease: 0.000
- min_samples_split: 3.000
- n_estimators: 75.000

### Importance des Caractéristiques pour Boosting

| Caractéristique | Importance (%) |
|----------------|----------------|
| matrix_size_x | 0.81 |
| matrix_size_y | 2.29 |
| p1 | 12.71 |
| p2 | 1.32 |
| p3 | 47.06 |
| p4 | 18.73 |
| p5 | 4.75 |
| p6 | 17.00 |
| p7 | -7.90 |
| p8 | 3.25 |

---

| Arbre de décision simple | 3.270e-04 | 0.323 | 0.001 |

### Paramètres de Arbre de décision simple

- max_depth: 60.000
- min_impurity_decrease: 0.000
- min_samples_split: 2.000

### Importance des Caractéristiques pour Arbre de décision simple

| Caractéristique | Importance (%) |
|----------------|----------------|
| matrix_size_x | -2.28 |
| matrix_size_y | 2.52 |
| p1 | 12.98 |
| p2 | 2.11 |
| p3 | 45.48 |
| p4 | 20.98 |
| p5 | 4.28 |
| p6 | 19.00 |
| p7 | -8.24 |
| p8 | 3.15 |

---

| XGBoost | 1.368e-03 | 1.326 | 0.002 |

### Paramètres de XGBoost

- gamma: 0.000
- lambda: 1.000
- learning_rate: 0.100
- max_depth: 5.000
- n_estimators: 10.000

### Importance des Caractéristiques pour XGBoost

| Caractéristique | Importance (%) |
|----------------|----------------|

---

### XGBoost

#### Performance Metrics
- MSE: 1.368e-03
- Training Time: 1.283 seconds
- Evaluation Time: 0.002 seconds

#### Model Parameters
- gamma: 0.000
- lambda: 1.000
- learning_rate: 0.100
- max_depth: 5.000
- n_estimators: 10.000

#### Feature Importance
Feature importance not available for this model.

---

### XGBoost

#### Performance Metrics
- MSE: 1.368e-03
- Training Time: 1.308 seconds
- Evaluation Time: 0.002 seconds

#### Model Parameters
- gamma: 0.000
- lambda: 1.000
- learning_rate: 0.100
- max_depth: 5.000
- n_estimators: 10.000

#### Feature Importance
Feature importance not available for this model.

---

### XGBoost

#### Performance Metrics
- MSE: 1.368e-03
- Training Time: 1.311 seconds
- Evaluation Time: 0.002 seconds

#### Model Parameters
- gamma: 0.000
- lambda: 1.000
- learning_rate: 0.100
- max_depth: 5.000
- n_estimators: 10.000

#### Feature Importance
Feature importance not available for this model.

---

### XGBoost

#### Performance Metrics
- MSE: 1.368e-03
- Training Time: 1.317 seconds
- Evaluation Time: 0.002 seconds

#### Model Parameters
- gamma: 0.000
- lambda: 1.000
- learning_rate: 0.100
- max_depth: 5.000
- n_estimators: 10.000

#### Feature Importance
- matrix_size_x: 4.31%
- matrix_size_y: 0.81%
- p1: 3.70%
- p2: 0.14%
- p3: 36.97%
- p4: 28.10%
- p5: 2.18%
- p6: 13.54%
- p7: 7.78%
- p8: 2.47%

---

### XGBoost

#### Performance Metrics
- MSE: 1.368e-03
- Training Time: 1.273 seconds
- Evaluation Time: 0.002 seconds

#### Model Parameters
- gamma: 0.000
- lambda: 1.000
- learning_rate: 0.100
- max_depth: 5.000
- n_estimators: 10.000

#### Feature Importance
- matrix_size_x: 4.31%
- matrix_size_y: 0.81%
- p1: 3.70%
- p2: 0.14%
- p3: 36.97%
- p4: 28.10%
- p5: 2.18%
- p6: 13.54%
- p7: 7.78%
- p8: 2.47%

---

