"""
train.py
--------
Trains the machine learning models for this project.

Two tasks:
  1. REGRESSION      -> predict Price_in_Lakhs (how much is this property worth?)
  2. CLASSIFICATION  -> predict Good_Investment (is it cheap for its city? 1 or 0)

Steps followed (the standard beginner ML workflow):
  1. Load the dataset
  2. Check missing values
  3. Fill missing values
  4. Convert text columns to numbers using pd.get_dummies()
  5. Separate X (features) and y (target)
  6. Split into training data and testing data
  7. Train the models
  8. Make predictions
  9. Evaluate the models

Run:  python train.py
"""

import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix

# Make the models folder if it does not exist yet
os.makedirs("models", exist_ok=True)

# =====================================================================
# STEP 1: Load the dataset
# =====================================================================
data = pd.read_csv("data/property_data.csv")
print("Rows:", len(data), " Columns:", len(data.columns))

# =====================================================================
# STEP 2: Check missing values
# =====================================================================
print("\nMissing values before cleaning:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# =====================================================================
# STEP 3: Fill missing values
# Numbers  -> filled with the median (middle value)
# Text     -> filled with the mode (most common value)
# =====================================================================
data["Nearby_Schools"] = data["Nearby_Schools"].fillna(data["Nearby_Schools"].median())
data["Furnished_Status"] = data["Furnished_Status"].fillna(
    data["Furnished_Status"].mode()[0]
)

print("\nMissing values after cleaning:", data.isnull().sum().sum())

# =====================================================================
# STEP 4: Convert text columns into numbers
# get_dummies turns "City" into City_Delhi, City_Mumbai, City_Pune ...
# with a 0 or 1 in each. Models can only work with numbers.
# =====================================================================
data_encoded = pd.get_dummies(data)
print("Columns after get_dummies:", len(data_encoded.columns))


# =====================================================================
# TASK 1: REGRESSION - predict the property price
# =====================================================================
print("\n" + "=" * 55)
print("TASK 1: REGRESSION - predicting Price_in_Lakhs")
print("=" * 55)

# STEP 5: Separate features (X) and target (y)
# We drop Good_Investment too, because it was calculated FROM the price.
# Using it here would be cheating (this is called data leakage).
X_price = data_encoded.drop(columns=["Price_in_Lakhs", "Good_Investment"])
y_price = data_encoded["Price_in_Lakhs"]

# STEP 6: Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_price, y_price, test_size=0.2, random_state=42
)
print("Training rows:", len(X_train), " Testing rows:", len(X_test))

# STEP 7 and 8: Train the models and make predictions
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

tree_model = DecisionTreeRegressor(max_depth=8, random_state=42)
tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

# STEP 9: Evaluate both models
print("\nLinear Regression:")
print("  MAE :", round(mean_absolute_error(y_test, linear_predictions), 2), "lakhs")
print("  RMSE:", round(mean_squared_error(y_test, linear_predictions) ** 0.5, 2), "lakhs")
print("  R2  :", round(r2_score(y_test, linear_predictions), 3))

print("\nDecision Tree Regressor:")
print("  MAE :", round(mean_absolute_error(y_test, tree_predictions), 2), "lakhs")
print("  RMSE:", round(mean_squared_error(y_test, tree_predictions) ** 0.5, 2), "lakhs")
print("  R2  :", round(r2_score(y_test, tree_predictions), 3))

# Keep whichever model has the higher R2 score
if r2_score(y_test, linear_predictions) > r2_score(y_test, tree_predictions):
    best_price_model = linear_model
    best_price_name = "Linear Regression"
else:
    best_price_model = tree_model
    best_price_name = "Decision Tree Regressor"

print("\nBetter model:", best_price_name)


# =====================================================================
# TASK 2: CLASSIFICATION - is this a good investment?
# =====================================================================
print("\n" + "=" * 55)
print("TASK 2: CLASSIFICATION - predicting Good_Investment")
print("=" * 55)

# STEP 5 again: features and target for the classification task
# Here the price IS allowed as a feature, because a buyer always knows
# the asking price of a property before deciding.
X_invest = data_encoded.drop(columns=["Good_Investment"])
y_invest = data_encoded["Good_Investment"]

# STEP 6: Split
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_invest, y_invest, test_size=0.2, random_state=42
)

# Extra step: put all the numbers on a similar scale.
# Price is in the thousands while BHK is only 1 to 5. Logistic Regression
# works much better when the numbers are on a similar scale.
# (A Decision Tree does not care either way, it only compares values,
#  so we scale both models and keep the code simple.)
scaler = StandardScaler()
X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)

# STEP 7 and 8: Train and predict
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train2_scaled, y_train2)
logistic_predictions = logistic_model.predict(X_test2_scaled)

tree_classifier = DecisionTreeClassifier(max_depth=8, random_state=42)
tree_classifier.fit(X_train2_scaled, y_train2)
tree_class_predictions = tree_classifier.predict(X_test2_scaled)

# STEP 9: Evaluate
logistic_accuracy = accuracy_score(y_test2, logistic_predictions)
tree_accuracy = accuracy_score(y_test2, tree_class_predictions)

print("\nLogistic Regression accuracy:", round(logistic_accuracy, 3))
print("Decision Tree accuracy      :", round(tree_accuracy, 3))

if logistic_accuracy > tree_accuracy:
    best_invest_model = logistic_model
    best_invest_name = "Logistic Regression"
    best_predictions = logistic_predictions
else:
    best_invest_model = tree_classifier
    best_invest_name = "Decision Tree Classifier"
    best_predictions = tree_class_predictions

print("\nBetter model:", best_invest_name)
print("\nConfusion matrix for", best_invest_name)
print(confusion_matrix(y_test2, best_predictions))
print("(rows = actual, columns = predicted, order: 0 then 1)")


# =====================================================================
# SAVE THE MODELS so the Streamlit app can use them
# joblib just writes a trained model to a file.
# We also save the column names, because the app must build its input
# in exactly the same column order that the models were trained on.
# =====================================================================
joblib.dump(best_price_model, "models/price_model.pkl")
joblib.dump(best_invest_model, "models/investment_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X_price.columns), "models/price_columns.pkl")
joblib.dump(list(X_invest.columns), "models/investment_columns.pkl")

print("\nModels saved in the models/ folder.")
print("Next step: streamlit run app.py")
