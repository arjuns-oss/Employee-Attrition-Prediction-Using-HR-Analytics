# Employee Attrition Prediction using HR Analytics
## Project Overview

This project focuses on analyzing employee data to predict attrition (whether an employee will leave the company or not) using machine learning techniques. By identifying key factors influencing attrition, organizations can take proactive steps to improve employee retention.

## Objectives
Predict employee attrition (Yes/No)
Identify important factors affecting attrition
Compare multiple machine learning models
Provide insights for HR decision-making
## Dataset

### Dataset Used:
IBM HR Analytics Employee Attrition Dataset
### Features
The dataset includes various employee-related attributes such as:
- Age
- Job Role
- Department
- Monthly Income
- Job Satisfaction
- Years at Company
- Overtime
- Work-Life Balance
### Target Variable
Attrition
Yes → Employee left the company
No → Employee stayed
## Technologies Used
Python
Pandas & NumPy (Data Processing)
Matplotlib & Seaborn (Visualization)
Scikit-learn (Machine Learning)
XGBoost (Advanced Model)
SHAP (Model Explainability)
## Methodology
1. Data Preprocessing
Handling missing values
Encoding categorical variables
Feature scaling
2. Exploratory Data Analysis (EDA)
Distribution analysis
Correlation heatmaps
Attrition patterns visualization
3. Feature Selection
Chi-Square Test
ANOVA (Analysis of Variance)
4. Model Building
Logistic Regression
Decision Tree
Random Forest
XGBoost
5. Model Evaluation
Accuracy
Precision
Recall
Confusion Matrix
6. Model Explainability
SHAP values used to interpret predictions
## Results & Comparison
Model	Accuracy
Logistic Regression	~90%
Decision Tree	~87%
KNN	~91%
Random Forest	~96%
XGBoost	~95%

Best Model: Random Forest (Highest Accuracy)
