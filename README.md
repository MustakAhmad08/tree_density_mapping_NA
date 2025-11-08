# Mapping Tree Density Across North America — Models (No Coordinates)

This repository provides four models to predict tree density (trees ha⁻¹) using harmonized
forest inventory plots and remote-sensing covariates.

**Confidentiality note:** Spatial coordinates (LAT/LON) are excluded from all scripts and outputs.

**Expected Input Files**
- `2025_20May_train.csv`
- `2025_20May_test.csv`
(both should include `N_y`, covariates, and categorical `bio6_binned`)

**Models**
- FFNN (PyTorch)
- Random Forest (scikit-learn)
- Ridge Regression (scikit-learn Pipeline)
- GLM + PCA (statsmodels)

**Usage**
```bash
python run_pipeline.py --model FFNN   # default
python run_pipeline.py --model RF
python run_pipeline.py --model RR
python run_pipeline.py --model GLM
```

Outputs: saved to `outputs/{MODEL}_predicted.csv` (only predicted + observed values).

---
Prepared for publication and GitHub archiving.

---

### 📊 Model Evaluation and Interpretation

Each model outputs three primary performance metrics:

| Metric | Description | Interpretation |
|:--------|:-------------|:----------------|
| **MAE (Mean Absolute Error)** | Average absolute difference between predicted and observed tree densities. | Indicates average prediction deviation (trees ha⁻¹); lower = better. |
| **RMSE (Root Mean Squared Error)** | Square root of mean squared residuals. | Penalizes large errors; sensitive to outliers. Lower = better model fit. |
| **R² (Coefficient of Determination)** | Fraction of variance in observed data explained by the model. | Measures explanatory power; closer to 1 = stronger predictive accuracy. |

#### Example Output (Random Forest)
```text
MAE: 310.52 | RMSE: 435.76 | R²: 0.392
Saved predictions to outputs/RF_predicted.csv
```

- **Interpretation:**  
  The model explains ~39% of spatial variation in tree density across held-out test regions,  
  with an average deviation of ~311 trees ha⁻¹.  

#### File Structure of Outputs

| File | Description |
|:------|:-------------|
| `outputs/FFNN_predicted.csv` | Deep learning predictions using feedforward neural network |
| `outputs/RF_predicted.csv` | Random forest baseline |
| `outputs/RR_predicted.csv` | Ridge regression with regularization |
| `outputs/GLM_predicted.csv` | PCA + OLS regression (statistical baseline) |

Each CSV contains:
```text
predicted, observed
523.4, 498.7
...
```
