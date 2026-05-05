# Project Title: HR Employee Attrition Analysis & Predictive Modeling

## Team Members:

| Name      | Reg No |
| :---------| :------|
| Arjun S   | 253105 |
| Gayathri  | [Reg N |
| Sreehari  | [Reg N |

---

## Problem Statement
Employee attrition poses a significant challenge to organizations, leading to increased recruitment costs and loss of institutional knowledge. By analyzing HR data, companies can identify patterns that lead to turnover. The objective of this project is to develop a predictive system that identifies employees at high risk of leaving using demographic, financial, and performance-related data. The project leverages statistical analysis and machine learning to provide actionable insights for HR departments.

The project follows the complete data science lifecycle:
*   **Data Preprocessing**
*   **Exploratory Data Analysis (EDA)**
*   **Feature Selection**
*   **Model Development**
*   **Evaluation**
*   **Deployment using Streamlit**

---

## Objectives
*   Identify key drivers of employee attrition, such as Overtime and Monthly Income.
*   Perform comprehensive exploratory data analysis to visualize workforce trends.
*   Build and compare multiple machine learning models to predict potential turnover.
*   Identify the best-performing model (XGBoost) for final deployment.
*   Deploy an interactive dashboard for HR managers to assess employee risk.

---

## Dataset
*   **Source:** IBM HR Analytics Employee Attrition & Performance.
*   **Features:** The dataset contains 35 features, including:
    *   **Demographics:** Age, Gender, Education, Marital Status.
    *   **Work History:** Years at Company, Job Role, Total Working Years.
    *   **Satisfaction:** Environment Satisfaction, Job Involvement, Work-Life Balance.
*   **Target Class:** `Attrition` (Yes/No).

---

## Methodology

### 1. Data Preprocessing
*   Removed redundant columns such as `EmployeeCount`, `Over18`, `StandardHours`, and `EmployeeNumber`.
*   Handled categorical variables using Label Encoding and One-Hot Encoding.
*   Addressed class imbalance between "Stayed" and "Left" records to ensure model fairness].

### 2. Exploratory Data Analysis (EDA)
*   Visualized the significant impact of **Overtime** on attrition rates.
*   Analyzed the correlation between **Monthly Income** levels and employee retention.
*   Generated heatmaps to identify multi-collinearity and relationships between features.

### 3. Feature Engineering
*   Selected top predictors using statistical tests like Chi-squared and ANOVA.
*   Identified "Overtime," "Monthly Income," and "Age" as the most significant features for the model.

### 4. Model Building
The following machine learning models were implemented and compared:
*   **Logistic Regression**
*   **Random Forest Classifier**
*   **XGBoost (Extreme Gradient Boosting)**

### 5. Model Evaluation
Models were evaluated based on:
*   Accuracy and F1-Score.
*   Precision and Recall to minimize false negatives (missing an employee who is actually leaving).
*   Confusion Matrix analysis.

---

## Results & Comparison

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| Logistic Regression | 0.84 | 0.82 |
| Random Forest | 0.86 | 0.85 |
| **XGBoost (Best Model)** | **0.89** | **0.88** |

### Best Model
**XGBoost** achieved the highest accuracy and F1-score. It was selected for final deployment because it effectively handles complex, non-linear relationships and demonstrated superior predictive power compared to traditional models.

---

## Model Performance Summary

### XGBoost
*   **Accuracy:** 89%, delivering the most reliable predictions for identifying at-risk employees.
*   **Strengths:** Excels at capturing intricate patterns in HR data and is highly efficient for real-time applications.

### Random Forest
*   **Accuracy:** 86%, providing solid performance and valuable feature importance insights.
*   **Strengths:** Less prone to overfitting and highly interpretable for identifying key turnover drivers.

### Logistic Regression
*   **Accuracy:** 84%, serving as a clear and interpretable baseline model.

---

## Conclusion
This project successfully identified the primary catalysts for employee turnover, with **Overtime** and **Low Monthly Income** being the most critical factors. By leveraging **XGBoost**, we achieved high predictive accuracy. The deployment of this system via **Streamlit** provides HR professionals with an interactive tool to assess employee risk in real-time, enabling proactive retention strategies.

---

## Deployment
The best-performing model is deployed using **Streamlit**, featuring:
*   **Manual Input:** Sliders and dropdowns for age, income, and satisfaction levels.
*   **Real-time Prediction:** Immediate calculation of attrition probability.
*   **Visual Analytics:** Interactive charts showing model performance and feature importance.

## Application Screenshots
### Home Interface
<img width="1918" height="820" alt="image" src="https://github.com/user-attachments/assets/029e1591-29fe-447b-aac6-78af4270bcba" />
### Model Performance & explainability
<img width="1569" height="746" alt="image" src="https://github.com/user-attachments/assets/baeb6746-5d72-48f8-892f-1adff8fe1067" />
### Model Information
<img width="684" height="251" alt="image" src="https://github.com/user-attachments/assets/d13c20ae-df59-45ef-ae0e-c4a45d3bb3f0" />



## Live Application
link: https://employee-attrition-prediction-using-hr-analytics-k2toifoqkxqhn.streamlit.app/
