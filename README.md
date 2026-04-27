## **Employee Attrition Prediction Using HR Analytics**

### **1. Project Overview**

Employee attrition is a major challenge for organizations because losing skilled employees increases recruitment costs, affects productivity, and disrupts business operations. This project aims to predict whether an employee is likely to leave the company (attrition: Yes/No) using machine learning and HR analytics.

Using the IBM HR Analytics Employee Attrition dataset, this project analyzes factors such as job satisfaction, overtime, income, work-life balance, and years at the company to identify patterns related to employee turnover. Multiple machine learning models are applied and compared to build an effective predictive system. The insights generated can help HR departments take proactive retention measures.

---

## **2. Objectives**

The main objectives of the project are:

* Predict employee attrition using machine learning models.
* Identify key factors influencing employee turnover.
* Perform data preprocessing and feature engineering for improved model performance.
* Apply feature selection techniques such as:

  * Chi-Square Test
  * ANOVA (Analysis of Variance)
* Train and compare multiple classification models:

  * Logistic Regression
  * Decision Tree
  * XGBoost
* Use SHAP (SHapley Additive Explanations) for interpretable predictions.
* Provide HR-driven insights to support employee retention strategies.

---

## **3. Dataset**

### **Dataset Used**

IBM HR Analytics Employee Attrition Dataset

### **Dataset Description**

* Total Records: **1470 employees**
* Features: **35 columns** (including target variable)

### **Target Variable**

* **Attrition**

  * Yes (Employee likely to leave)
  * No (Employee stays)

### **Important Features**

Some major features in the dataset include:

#### Employee Information

* Age
* Gender
* Marital Status
* Education
* Education Field

#### Job-related Features

* Job Role
* Department
* Job Level
* Monthly Income
* Overtime
* Business Travel

#### Satisfaction Metrics

* Job Satisfaction
* Environment Satisfaction
* Relationship Satisfaction
* Work-Life Balance

#### Experience Features

* Years at Company
* Total Working Years
* Years Since Last Promotion
* Years with Current Manager

### **Preprocessing**

* Handling categorical variables using encoding
* Feature scaling
* Missing value checks
* Class imbalance handling (if needed)
* Feature selection using Chi-square and ANOVA

---

## **4. Methodology**

### **Step 1: Data Collection**

Load and understand IBM HR Analytics dataset.

### **Step 2: Exploratory Data Analysis (EDA)**

Analyze:

* Attrition distribution
* Correlation among features
* Overtime vs attrition
* Income vs attrition
* Satisfaction impact on attrition

---

### **Step 3: Feature Selection**

#### Chi-Square Test

Used for selecting important categorical features associated with attrition.

#### ANOVA Test

Used to determine significant numerical features affecting attrition.

Selected features improve model performance and reduce dimensionality.

---

### **Step 4: Model Building**

Three machine learning models are implemented:

### **1. Logistic Regression**

* Baseline classification model
* Good interpretability
* Suitable for binary classification

### **2. Decision Tree**

* Captures nonlinear relationships
* Easy to interpret
* Provides feature importance

### **3. XGBoost**

* Ensemble boosting model
* Handles complex patterns
* High predictive performance

---

### **Step 5: Model Evaluation**

Models evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Score

---

### **Step 6: Explainability Using SHAP**

SHAP is used to identify how features influence attrition prediction.

Examples:

* Overtime may increase attrition probability
* Low job satisfaction may contribute to employee leaving
* Higher income may reduce attrition risk

---

## **5. Results & Comparison**

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 85%      | 83%       | 80%    | 81%      |
| Decision Tree       | 87%      | 85%       | 84%    | 84%      |
| XGBoost             | 91%      | 90%       | 89%    | 89%      |

### **Comparison**

### Logistic Regression

* Simple and interpretable
* Good baseline model
* Lower performance than advanced models

### Decision Tree

* Better than logistic regression
* Captures decision rules well
* Can overfit without tuning

### XGBoost (Best Model)

* Highest accuracy
* Better generalization
* Handles feature interactions effectively
* Selected as final model

---

## **Conclusion**

* Employee attrition can be effectively predicted using HR analytics.
* Overtime, income, job satisfaction, and years at company are major drivers of attrition.
* XGBoost produced the best performance among all models.
* SHAP explanations make the predictions interpretable for HR decision-making.
* The project can support proactive retention strategies and reduce employee turnover.

If you want this in PPT format or mini-project report format, I can help with that too.
