"""
eda.py
------
EDA means Exploratory Data Analysis: looking at the data with tables and
charts BEFORE building any model, so we understand what we are working with.

This file answers 6 questions. Each one has a clear purpose and each chart
is saved as a PNG in the outputs/ folder.

Run:  python eda.py
"""

import os

import matplotlib
matplotlib.use("Agg")          # save charts as files instead of opening windows

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Make the outputs folder if it does not exist yet
os.makedirs("outputs", exist_ok=True)

data = pd.read_csv("data/property_data.csv")
print("Rows:", len(data), " Columns:", len(data.columns))
print("\nFirst 5 rows:")
print(data.head())

print("\nBasic statistics:")
print(data.describe().round(2))

print("\nMissing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])


# =====================================================================
# Q1: How are property prices spread out?
# Purpose: see the typical price and whether a few expensive properties
#          are stretching the range.
# =====================================================================
print("\n--- Q1. How are property prices spread out?")
print(data["Price_in_Lakhs"].describe().round(2))

plt.figure(figsize=(7, 4))
plt.hist(data["Price_in_Lakhs"], bins=40, color="steelblue", edgecolor="white")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Number of properties")
plt.title("Q1: Distribution of property prices")
plt.tight_layout()
plt.savefig("outputs/q1_price_distribution.png", dpi=100)
plt.close()


# =====================================================================
# Q2: Does a bigger property cost more?
# Purpose: check if Size is a useful feature for predicting price.
# =====================================================================
print("\n--- Q2. Does a bigger property cost more?")
correlation = data["Size_in_SqFt"].corr(data["Price_in_Lakhs"])
print("Correlation between size and price:", round(correlation, 3))

plt.figure(figsize=(6, 4))
plt.scatter(data["Size_in_SqFt"], data["Price_in_Lakhs"], s=8, alpha=0.4)
plt.xlabel("Size (SqFt)")
plt.ylabel("Price (Lakhs)")
plt.title("Q2: Size vs Price")
plt.tight_layout()
plt.savefig("outputs/q2_size_vs_price.png", dpi=100)
plt.close()


# =====================================================================
# Q3: Which city is the most expensive?
# Purpose: confirm that City matters, so it must be included as a feature.
# =====================================================================
print("\n--- Q3. Which city is the most expensive?")
average_price_by_city = (
    data.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False)
)
print(average_price_by_city.round(1))

plt.figure(figsize=(7, 4))
average_price_by_city.plot(kind="bar", color="indianred")
plt.ylabel("Average price (Lakhs)")
plt.title("Q3: Average property price by city")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("outputs/q3_price_by_city.png", dpi=100)
plt.close()


# =====================================================================
# Q4: Does furnishing change the price?
# Purpose: decide whether Furnished_Status is worth keeping as a feature.
# =====================================================================
print("\n--- Q4. Does furnishing change the price?")
print(data.groupby("Furnished_Status")["Price_in_Lakhs"].median().round(1))

plt.figure(figsize=(7, 4))
sns.boxplot(data=data, x="Furnished_Status", y="Price_in_Lakhs")
plt.title("Q4: Price by furnished status")
plt.tight_layout()
plt.savefig("outputs/q4_price_by_furnishing.png", dpi=100)
plt.close()


# =====================================================================
# Q5: How are the numeric columns related to each other?
# Purpose: spot which numbers move together, and catch any column that
#          would give the answer away.
# =====================================================================
print("\n--- Q5. How are the numeric columns related?")
numeric_columns = ["BHK", "Size_in_SqFt", "Age_of_Property",
                   "Nearby_Schools", "Nearby_Hospitals", "Price_in_Lakhs"]
correlation_table = data[numeric_columns].corr()
print(correlation_table.round(2))

plt.figure(figsize=(7, 5))
sns.heatmap(correlation_table, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Q5: Correlation heatmap")
plt.tight_layout()
plt.savefig("outputs/q5_correlation_heatmap.png", dpi=100)
plt.close()


# =====================================================================
# Q6: How balanced is the Good_Investment label?
# Purpose: if one class were much bigger, accuracy would be misleading.
# =====================================================================
print("\n--- Q6. How balanced is the Good_Investment label?")
label_counts = data["Good_Investment"].value_counts().sort_index()
print(label_counts)
print("Percentage that are good investments:",
      round(data["Good_Investment"].mean() * 100, 1), "%")

plt.figure(figsize=(5, 4))
plt.bar(["Not good (0)", "Good (1)"], label_counts.values, color=["grey", "seagreen"])
plt.ylabel("Number of properties")
plt.title("Q6: Balance of the Good_Investment label")
plt.tight_layout()
plt.savefig("outputs/q6_label_balance.png", dpi=100)
plt.close()


print("\nAll 6 charts saved in the outputs/ folder.")
