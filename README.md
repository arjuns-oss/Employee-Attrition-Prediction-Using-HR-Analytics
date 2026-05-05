# Project Title: HR Employee Attrition Analysis & Predictive Modeling

**Team Members:**

| Name      | Reg No |
| :---------| :------|
| Arjun S   | 253105 |
| Gayathri  | [Reg N |
| Sreehari  | [Reg N |

---

## Problem Statement
Employee attrition poses a significant challenge to organizations, leading to increased recruitment costs and loss of institutional knowledge[cite: 1]. By analyzing HR data, companies can identify patterns that lead to turnover[cite: 1]. The objective of this project is to develop a predictive system that identifies employees at high risk of leaving using demographic, financial, and performance-related data[cite: 1]. The project leverages statistical analysis and machine learning to provide actionable insights for HR departments[cite: 1].

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
*   **Source:** IBM HR Analytics Employee Attrition & Performance (via Kaggle).
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
*   Visualized the significant impact of **Overtime** on attrition rates[cite: 1].
*   Analyzed the correlation between **Monthly Income** levels and employee retention[cite: 1].
*   Generated heatmaps to identify multi-collinearity and relationships between features[cite: 1].

### 3. Feature Engineering
*   Selected top predictors using statistical tests like Chi-squared and ANOVA[cite: 1].
*   Identified "Overtime," "Monthly Income," and "Age" as the most significant features for the model[cite: 1].

### 4. Model Building
The following machine learning models were implemented and compared:
*   **Logistic Regression**[cite: 1]
*   **Random Forest Classifier**[cite: 1]
*   **XGBoost (Extreme Gradient Boosting)**[cite: 1]

### 5. Model Evaluation
Models were evaluated based on:
*   Accuracy and F1-Score[cite: 1].
*   Precision and Recall to minimize false negatives (missing an employee who is actually leaving)[cite: 1].
*   Confusion Matrix analysis[cite: 1].

---

## Results & Comparison

| Model | Accuracy | F1-Score |
| :--- | :--- | :--- |
| Logistic Regression | 0.84 | 0.82 |
| Random Forest | 0.86 | 0.85 |
| **XGBoost (Best Model)** | **0.89** | **0.88** |

### Best Model
**XGBoost** achieved the highest accuracy and F1-score[cite: 1]. It was selected for final deployment because it effectively handles complex, non-linear relationships and demonstrated superior predictive power compared to traditional models[cite: 1].

---

## Model Performance Summary

### XGBoost
*   **Accuracy:** 89%, delivering the most reliable predictions for identifying at-risk employees[cite: 1].
*   **Strengths:** Excels at capturing intricate patterns in HR data and is highly efficient for real-time applications[cite: 1].

### Random Forest
*   **Accuracy:** 86%, providing solid performance and valuable feature importance insights[cite: 1].
*   **Strengths:** Less prone to overfitting and highly interpretable for identifying key turnover drivers[cite: 1].

### Logistic Regression
*   **Accuracy:** 84%, serving as a clear and interpretable baseline model[cite: 1].

---

## Conclusion
This project successfully identified the primary catalysts for employee turnover, with **Overtime** and **Low Monthly Income** being the most critical factors[cite: 1]. By leveraging **XGBoost**, we achieved high predictive accuracy[cite: 1]. The deployment of this system via **Streamlit** provides HR professionals with an interactive tool to assess employee risk in real-time, enabling proactive retention strategies[cite: 1].

---

## Deployment
The best-performing model is deployed using **Streamlit**, featuring:
*   **Manual Input:** Sliders and dropdowns for age, income, and satisfaction levels[cite: 1].
*   **Real-time Prediction:** Immediate calculation of attrition probability[cite: 1].
*   **Visual Analytics:** Interactive charts showing model performance and feature importance[cite: 1].

### link: https://employee-attrition-prediction-using-hr-analytics-k2toifoqkxqhn.streamlit.app/
