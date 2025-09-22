# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease using patient data. Implements **Logistic Regression** and provides visualizations to explore feature relationships.

---

## Table of Contents
- Overview
- Dataset
- Features
- Installation
- Usage
- Model
- Evaluation
- Example Prediction
- Visualization
- License

---

## Overview

This project uses patient health data to predict heart disease risk. It includes preprocessing, feature engineering, train-test splitting, model training with Logistic Regression, evaluation, and visualization of key features.  

A new feature, **cholesterol-to-age ratio**, is added to enhance prediction performance.

---

## Dataset

The dataset used is `heart.csv`. It includes patient information such as:

- age
- sex
- chest pain type (`cp`)
- resting blood pressure (`trestbps`)
- serum cholesterol (`chol`)
- fasting blood sugar (`fbs`)
- resting ECG results (`restecg`)
- maximum heart rate achieved (`thalach`)
- exercise-induced angina (`exang`)
- ST depression induced by exercise (`oldpeak`)
- slope of peak exercise ST segment (`slope`)
- number of major vessels (`ca`)
- thalassemia (`thal`)
- target (heart disease presence: 0 = No, 1 = Yes)

---

## Features

- **Standard patient features**: age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Engineered feature**: `chol_age_ratio` = cholesterol / age
- **Target variable**: `target` (0: No heart disease, 1: Heart disease)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/shaunak-alt/heart-disease-prediction.git
```

2. Install required Python packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## Usage

1. Ensure the dataset `heart.csv` is in the same folder as the script or update the file path:

```
file_path = 'heart.csv'
```

2. Run the Python script:

```bash
python heart-disease.py
```

The script performs:

- Data loading and exploration
- Feature engineering (`chol_age_ratio`)
- Train-test split
- Logistic Regression model training
- Evaluation on training and test sets
- Example prediction for new patient data
- Visualization of feature distributions

---

## Model

**Algorithm:** Logistic Regression  
**Max Iterations:** 1000  

Alternative models like `RandomForestClassifier` can be substituted if desired.

---

## Evaluation

The model is evaluated using:

- **Accuracy Score** on training and test data
- **Classification Report** (precision, recall, F1-score)
- **Confusion Matrix**

---

## Example Prediction

Example for a new patient:

```
input_data = (
63, # age
1, # sex
3, # cp
145, # trestbps
233, # chol
1, # fbs
0, # restecg
150, # thalach
0, # exang
2.3, # oldpeak
0, # slope
0, # ca
1, # thal
chol/age # chol_age_ratio
)
prediction = model.predict(input_data_array)
```


Outputs whether the person **HAS** or **does NOT have** heart disease.

---

## Visualization

The script provides:

- Histogram of `chol_age_ratio` by heart disease status
- Boxplot of `chol_age_ratio` by heart disease status

Helps explore the relationship between cholesterol-to-age ratio and heart disease.
