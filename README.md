# 🏠 Real Estate Investment Advisor

A beginner-level Machine Learning project that analyzes simulated Indian real estate data to predict property prices and identify potentially good investment opportunities.

## 📌 About the Project

Buying a property involves two important questions:

1. 💰 **What is the estimated price of the property?**
2. 🏠 **Is the property a potentially good investment compared to other properties in the same city?**

This project uses basic Machine Learning techniques to answer these questions.

The project includes:

- Data generation and preprocessing
- Exploratory Data Analysis (EDA)
- Regression for property price prediction
- Classification for investment prediction
- Model evaluation
- A simple Streamlit web application

## 🎯 Objectives

- Analyze property price and size patterns
- Understand how location and property features affect price
- Predict property prices using Regression
- Classify properties as potentially good or not good investments
- Build a simple interactive web application

## 🧠 Machine Learning Tasks

### 1. Property Price Prediction

**Type:** Regression

The model predicts the property's price in lakhs.

**Target:**
`Price_in_Lakhs`

### 2. Investment Prediction

**Type:** Classification

The model predicts whether a property is considered a potentially good investment based on its price per square foot compared with the median price in its city.

**Target:**
`Good_Investment`

## 📊 Dataset

The dataset contains **2,000 simulated property records**.

Some of the features include:

- City
- Property Type
- BHK
- Size in SqFt
- Age of Property
- Furnishing Status
- Public Transport
- Nearby Schools
- Nearby Hospitals
- Parking
- Security
- Price in Lakhs

> **Note:** The dataset is simulated for educational purposes and does not represent actual real estate market data.

## 🔍 Exploratory Data Analysis

The project performs six basic EDA questions:

1. How are property prices distributed?
2. Does a bigger property cost more?
3. Which city has the highest average property price?
4. Does furnishing affect property price?
5. How are numerical features related?
6. How balanced is the Good Investment classification?

The analysis uses simple charts and statistical summaries to understand the dataset before training the models.

## 🤖 Machine Learning Models

The project uses beginner-friendly models:

### Regression
- Linear Regression
- Decision Tree Regressor

### Classification
- Logistic Regression
- Decision Tree Classifier

These models are compared using basic evaluation metrics.

## 📈 Model Evaluation

### Classification

- Accuracy
- Confusion Matrix

### Regression

- MAE
- RMSE
- R² Score

## 🌐 Streamlit Application

The project includes a simple Streamlit application with three sections:

- **Home** – Project overview
- **Dataset** – Basic dataset information and visualizations
- **Prediction** – Enter property details and receive predictions

Run the application with:

```bash
streamlit run app.py
