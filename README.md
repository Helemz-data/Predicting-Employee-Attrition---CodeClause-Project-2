HR Attrition Prediction System
Overview

This project develops a machine learning solution that predicts the likelihood of employee attrition using HR analytics data. It includes data preprocessing, exploratory data analysis (EDA), model development, evaluation, and deployment of an interactive dashboard for HR decision-makers.

The system assists organizations in identifying at-risk employees and proactively designing retention strategies.

Key Features

End-to-end data science pipeline

HR dashboard built with Dash

Multiple classification models (Logistic Regression, Random Forest, Gradient Boosting, etc.)

Feature importance visualization

Clean and modular code structure

GitHub-ready project layout

Project Structure
hr-attrition-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── utils.py
│
├── models/
│   └── attrition_model.pkl
│
├── app/
│   ├── dashboard.py
│   └── callbacks.py
│
├── reports/
│   ├── final_report.pdf
│   └── figures/
│
├── tests/
│   └── test_preprocessing.py
│
├── README.md
├── requirements.txt
└── .gitignore

Objective

To build a robust classification model that predicts if an employee is likely to leave the company.

Dataset

The dataset includes employee information such as:

Demographics

Job role and department

Compensation

Performance rating

Work environment metrics

Overtime status

Job satisfaction

The target variable is:

Attrition: Yes/No

Methodology
1. Data Preprocessing

Handling missing values

Label encoding and one-hot encoding

Scaling numeric features

SMOTE for class imbalance (if applied)

Train-test split

2. Exploratory Data Analysis (EDA)

Attrition distribution

Correlation heatmap

Income vs attrition

Job satisfaction patterns

Feature relationships

3. Model Development

Several models were trained:

Logistic Regression

Random Forest

Gradient Boosting Classifier

Hyperparameter tuning using RandomizedSearchCV improved model performance.

4. Model Evaluation

Metrics monitored:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Typical best model performance:

Accuracy: ~85–90%

ROC-AUC: ~0.88–0.92

Dashboard Application

The interactive dashboard (built with Dash) includes:

Attrition insights

Feature importance charts

Data exploration tools

Prediction interface

Results

Overtime strongly increases attrition risk

Low income and low job satisfaction correlate with higher turnover

Models are effective in identifying at-risk employees

Dashboard supports HR strategic planning

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Plotly

Dash

Matplotlib / Seaborn (EDA)

