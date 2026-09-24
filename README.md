# 🚢 Titanic Survival Prediction & Machine Learning Benchmark

**Author:** Sehar Naeem  
**Internship:** Inovegen AI/ML Internship Program 2026  
**Task:** Task 2 — Supervised Classification Model & Evaluation  
**Interactive Colab Notebook:** [View in Google Colab](https://colab.research.google.com/drive/1uUbUrOpCM9oK1TchMywEzScAIOHSxNpa?usp=sharing)  

---

## 📌 Project Overview
This project develops, evaluates, and benchmarks supervised machine learning models to predict passenger survival on the Titanic (`Survived`: 0 = Perished, 1 = Survived). 

The pipeline spans the entire machine learning lifecycle:
1. **Data Preprocessing & Quality Assurance:** Handling missing values and encoding categorical features.
2. **Train/Test Splitting & Standardization:** Stratified 80/20 train-test division and z-score feature scaling.
3. **Model Training & Comparison:** Benchmarking a linear baseline (**Logistic Regression**) against a non-linear ensemble (**Random Forest Classifier**).
4. **Diagnostic Visualizations:** Confusion Matrix, ROC-AUC Curve, and Feature Importance ranking.
5. **Interactive Model Deployment:** Live simulation of custom passenger profiles (Jack vs. Rose scenario).
6. **Cross-Validation & Rigorous Reporting:** 5-Fold Cross-Validation and complete Precision/Recall/F1 metrics.

---

## 🏆 Model Performance Benchmark

| Evaluation Metric | Logistic Regression (Baseline) | Random Forest Classifier (Alternative) | Best Performing |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **81.01%** | 79.89% | **Logistic Regression** |
| **5-Fold Cross-Validation (Mean)** | **78.93% (+/- 2.83%)** | 79.10% | **Consistent / Robust** |
| **ROC-AUC Score** | **0.88** | 0.84 | **Logistic Regression** |
| **Precision (Survivors)** | **0.80** | 0.78 | **Logistic Regression** |
| **Recall (Survivors)** | **0.70** | 0.69 | **Logistic Regression** |
| **F1-Score (Survivors)** | **0.75** | 0.73 | **Logistic Regression** |

> **Key Finding:** On this dataset (~891 records), the linear model (Logistic Regression) slightly outperformed the complex ensemble model (Random Forest), demonstrating superior resistance to overfitting on small sample sizes.

---

## 📊 Visualizations & Key Findings

### 1. Confusion Matrix Heatmap
- **True Negatives:** 90 non-survivors correctly identified.
- **True Positives:** 52 survivors correctly caught.
- **Low False Alarm Rate:** High specificity with balanced precision and recall.

### 2. ROC-AUC Diagnostic Curve
- Achieved an **AUC of 0.88**, significantly exceeding the 0.50 random-guessing baseline. This confirms the classifier maintains high sensitivity across varying probability decision thresholds.

### 3. Feature Importance Analysis
The Random Forest feature importance analysis revealed the top 3 drivers of survival:
1. **`Fare` (0.274):** Direct indicator of upper-deck proximity and lifeboat access.
2. **`Sex` (0.268):** Reflected the strict historical adherence to the "women and children first" protocol.
3. **`Age` (0.253):** Survival prioritization for pediatric passenger demographics.

### 4. Live Custom Passenger Simulation
To test the practical inference capabilities of the model, two hypothetical passengers were evaluated:
* **Rose Dawson (1st Class Female, 20 yrs, $150 Fare):** Predicted **SURVIVED** (Survival Probability: **95.4%**)
* **Jack Dawson (3rd Class Male, 20 yrs, $8 Fare):** Predicted **PERISHED** (Survival Probability: **12.9%**)

---


---

## 💡 Conclusions, Limitations & Next Steps

### Limitations
1. **Sample Size:** 891 records is small for high-dimensional models.
2. **Median Imputation:** Preserves records, but dampens age variance.
3. **Cabin Omission:** >77% missing values forced dropping cabin deck data, despite its historical relevance to evacuation speed.

### Future Recommendations
* **Feature Engineering:** Extract titles (*Mr., Mrs., Miss, Master, Dr.*) from the `Name` field, and engineer a composite `FamilySize` feature (`SibSp + Parch + 1`).
* **Hyperparameter Optimization:** Conduct `GridSearchCV` on Random Forest hyperparameters (`max_depth`, `min_samples_split`, `criterion`).
* **Advanced Ensembles:** Benchmark gradient boosting architectures (**XGBoost** and **LightGBM**).

---

## 💻 Tech Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebook

