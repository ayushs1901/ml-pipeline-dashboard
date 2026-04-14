# 🚀 AutoML Pipeline Dashboard

An **interactive end-to-end Machine Learning pipeline** built using **Streamlit**, designed to simplify the complete ML workflow — from data upload to model tuning — all in one place.

This project provides a **user-friendly dashboard** where users can perform data preprocessing, visualization, model training, and optimization without writing complex code.

---

## 📌 Project Overview

This application demonstrates a **complete Machine Learning lifecycle**, including:

- 📂 Data Upload & Configuration  
- 📊 Exploratory Data Analysis (EDA)  
- 🛠 Data Cleaning & Engineering  
- 🎯 Feature Selection  
- 🤖 Model Training & Validation  
- ⚙️ Hyperparameter Tuning  

All steps are organized in a **horizontal pipeline structure**, making it intuitive and visually appealing.

---

## ✨ Features

### 🔹 1. Problem Selection
- Supports both:
  - Classification
  - Regression

---

### 🔹 2. Data Input
- Upload CSV or Excel files
- Select target variable
- Choose input features dynamically

---

### 🔹 3. Exploratory Data Analysis (EDA)
- Dataset summary (mean, std, etc.)
- Missing value analysis
- Correlation heatmap
- PCA-based data visualization (2D projection)

---

### 🔹 4. Data Cleaning & Engineering
- Missing value imputation:
  - Mean
  - Median
  - Mode
- Outlier detection methods:
  - IQR
  - Isolation Forest
  - DBSCAN
  - OPTICS
- Option to remove outliers directly from UI

---

### 🔹 5. Feature Selection
- Variance Threshold
- Mutual Information (Information Gain)
- Correlation-based insights

---

### 🔹 6. Model Selection
Supports multiple models:
- Linear Regression / Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Means Clustering

---

### 🔹 7. Model Training & Validation
- Train-Test Split
- K-Fold Cross Validation
- Performance metrics:
  - Accuracy (Classification)
  - RMSE / R² (Regression)
- Overfitting & Underfitting detection

---

### 🔹 8. Hyperparameter Tuning
- Grid Search
- Randomized Search
- Displays best parameters and improved scores

---

## 🛠 Tech Stack

- **Frontend/UI**: Streamlit  
- **Data Handling**: Pandas, NumPy  
- **Visualization**: Plotly  
- **Machine Learning**: Scikit-learn  

---

## 📂 Project Structure
📁 ml-pipeline-dashboard
│
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
└── README.md # Project documentation
