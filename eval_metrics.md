# Metrics of the model **CatBoostRegressor**

---

### Selected Indicators
The model used these 5 features:
1. **Livable Space (m²)**
2. **Avg_Taxable_Income**
3. **Province**
4. **State of the Building**
5. **Subtype of Property_Grouped**

The target variable, **Price**, was logarithmized to reduce the influence of extreme values, stabilize the model, and approximate a normal distribution. No standardization was needed as tree-based models are invariant to scaling.

### Training Configuration
- **Categorical variables:** 'Province,' 'State of the Building,' 'Subtype of Property_Grouped.'
- **Train-test split ratio:** 80-20%.
- **Training parameters:** 
  - Iterations = 1000
  - Learning rate = 0.05
  - Depth = 6

---

## Model Evaluation

![alt text](output_data/image-2.png)

### Performance on Original Data:
- **Train Set:**
  - **R² = 0.8167**
  - **RMSE = 151,856.91**
  - **MAE = 96,608.86**
  - **MAPE = 17.38%**
  - **sMAPE = 17.08%**
- **Test Set:**
  - **R² = 0.7823**
  - **RMSE = 158,580.73**
  - **MAE = 104,455.30**
  - **MAPE = 19.45%**
  - **sMAPE = 18.93%**

---

## Conclusion

- **Better Performance on Log-Transformed Data:** The log transformation reduces the effect of extreme values, allowing the model to perform more accurately. Evaluating Metrics are consistently better in this space.
- **Challenges on the Original Scale:** The model struggles to handle the wide range and skewed distribution of the original data. High RMSE and MAE indicate difficulty predicting very large prices.
- **Generalization Gap:** Slightly lower performance on test data compared to training data suggests some level of overfitting. This might be mitigated by regularization or enhanced cross-validation.

### Execution Time
The training code execution took approximately **50 seconds**.
