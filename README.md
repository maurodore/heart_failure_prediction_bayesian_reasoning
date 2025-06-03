# Bayesian Heart Failure Prediction

## 📜 Abstract

This project implements and evaluates Bayesian Networks (BNs) for predicting heart failure using a dataset of 918 patient records with 11 clinical features. Four BN modeling strategies were developed and compared: a custom-built model, a TreeSearch-derived model, and two HillClimbSearch-derived models (BDeu score), one with and one without structural constraints. The primary goals were to assess predictive accuracy, computational efficiency, and derive insights into cardiovascular disease (CVD) risk factors.

---

## 📊 Dataset

* **Source**: "Heart Failure Prediction" dataset from Kaggle ([fedesoriano 2021](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)).
* **Features**: 11 relevant attributes including demographic data, clinical measurements (e.g., resting blood pressure, serum cholesterol), and electrocardiographic (ECG) findings (e.g., resting ECG results, ST slope).

---

## 🎯 Objectives

* Develop Bayesian Network models for CVD risk assessment.
* Compare four BN modeling strategies:
    * Custom-built (domain knowledge).
    * Search-based: TreeSearch.
    * Search-based: HillClimbSearch (BDeu score).
    * Search-based with constraint: HillClimbSearch (BDeu score) with domain-specific constraints and node pruning.
* Evaluate models based on predictive accuracy and computational runtime.
* Investigate clinical conditions leading to heart disease through model queries.

---

## ⚙️ Methodology

* **Library**: `pgmpy` for Bayesian Network construction and analysis.
* **Data Preprocessing**:
    * Label encoding for categorical features.
    * Discretization of continuous features based on medical guidelines.
    * Imputation of missing values using the median.
* **Data Splitting**: 80% for training, 20% for testing.
* **Model Structures**:
    * **Custom Model**: Hand-crafted based on cardiology knowledge and data correlation.
    * **Tree Model**: Learned via TreeSearch algorithm, all features, no structural constraints.
    * **HillClimb Model**: Learned via HillClimbSearch, all features, no structural constraints.
    * **Constraint-Based Model**: HillClimbSearch with enforced constraints and pruning of low-correlation nodes.
* **Parameter Learning**: Conditional Probability Tables (CPTs) learned using Maximum Likelihood Estimation from training data.
* **Inference**: Variable Elimination algorithm applied on the test set.
* **Scoring Functions**: BDeuScore primarily used for optimization; K2Score also tested.

---

## 📈 Results

### Performance Metrics

* **Accuracy**:
    * HillClimbSearch models achieved the best performance scores.
    * Constraint-Based HillClimb model: >87% accuracy.
    * All models: >84% accuracy.
    * TreeSearch model: Moderate performance.
    * Custom-built model: Worst accuracy.
* **Runtime Efficiency**:
    * TreeSearch model: Most efficient.
    * HillClimb model with constraints: Second most efficient.
    * Full HillClimb model and Custom model: Least efficient (Custom model had the longest execution times).

### Key Query Findings

* **Sex and Age as Risk Factors**:
    * Men are significantly more at risk of heart disease than women across all models.
    * Older individuals are more prone to CVDs.
* **Stress Test Indicators and CVD**:
    * Confirmed link between under-stress symptoms and CVDs.
    * Flat and Downsloping ST segment patterns are associated with Heart Disease.
    * Exercise-induced Angina is a strong indicator of coronary artery disease.
    * High Oldpeak values suggest ischemia.
* **CVD in Young Adults**:
    * Among young people, absence of angina during exercise does not exclude CVDs.
    * The asymptomatic young group is at the highest risk.

---

## 🏁 Conclusion

Bayesian Networks were effectively applied for CVD diagnosis and explanation using clinical data. The HillClimb model with constraints demonstrated the best balance of high accuracy (>87%) and good computational efficiency. Further testing on new data and expert medical consultation for refining constraints are recommended to enhance model robustness and potential.
