"""
make_dataset.py
---------------
Creates the dataset used by this project: data/property_data.csv

IMPORTANT (say this in the viva):
This is a SIMULATED dataset. It is not real market data. I generated it
myself so the project can run without a paid data source.

How the price is built:
    price per sqft = city base rate x furnishing effect x age effect
                     x transport effect x random noise
    price in lakhs = price per sqft x size / 100000

Because of the random noise, no model can be 100% correct. That is on
purpose - it makes the machine learning part honest.

Run:  python make_dataset.py
"""

import os

import numpy as np
import pandas as pd

# Make the data folder if it does not exist yet
os.makedirs("data", exist_ok=True)

# Same seed every time, so everyone gets the same dataset
np.random.seed(42)

NUMBER_OF_ROWS = 2000

# Average price per square foot in each city (in rupees)
CITY_BASE_RATE = {
    "Mumbai": 22000,
    "Delhi": 15000,
    "Bangalore": 9500,
    "Pune": 8500,
    "Jaipur": 4500,
    "Indore": 4000,
}

cities = list(CITY_BASE_RATE.keys())

# ---------- Step 1: create the basic property details ----------
city = np.random.choice(cities, NUMBER_OF_ROWS)
size = np.random.normal(1250, 400, NUMBER_OF_ROWS).round()
size = np.clip(size, 400, 3500)

bhk = np.clip((size / 480).round(), 1, 5).astype(int)
age = np.random.randint(0, 41, NUMBER_OF_ROWS)

property_type = np.random.choice(
    ["Apartment", "Villa", "Independent House"],
    NUMBER_OF_ROWS,
    p=[0.65, 0.15, 0.20],
)
furnished = np.random.choice(
    ["Unfurnished", "Semi-Furnished", "Fully-Furnished"],
    NUMBER_OF_ROWS,
    p=[0.45, 0.35, 0.20],
)
transport = np.random.choice(
    ["Low", "Medium", "High"], NUMBER_OF_ROWS, p=[0.25, 0.45, 0.30]
)

schools = np.random.randint(0, 11, NUMBER_OF_ROWS)
hospitals = np.random.randint(0, 9, NUMBER_OF_ROWS)
parking = np.random.choice(["Yes", "No"], NUMBER_OF_ROWS, p=[0.7, 0.3])
security = np.random.choice(["Yes", "No"], NUMBER_OF_ROWS, p=[0.65, 0.35])

# ---------- Step 2: build the price per square foot ----------
base_rate = np.array([CITY_BASE_RATE[c] for c in city])

furnish_effect = pd.Series(furnished).map(
    {"Unfurnished": 1.00, "Semi-Furnished": 1.05, "Fully-Furnished": 1.12}
).values

# Older buildings are cheaper: about 0.6% less per year
age_effect = 1 - (age * 0.006)

transport_effect = pd.Series(transport).map(
    {"Low": 0.95, "Medium": 1.00, "High": 1.08}
).values

# Random noise: two identical flats do not cost exactly the same
noise = np.random.normal(1.0, 0.15, NUMBER_OF_ROWS).clip(0.6, 1.5)

price_per_sqft = base_rate * furnish_effect * age_effect * transport_effect * noise
price_in_lakhs = (price_per_sqft * size / 100000).round(2)

# ---------- Step 3: put everything in a table ----------
data = pd.DataFrame({
    "City": city,
    "Property_Type": property_type,
    "BHK": bhk,
    "Size_in_SqFt": size.astype(int),
    "Age_of_Property": age,
    "Furnished_Status": furnished,
    "Public_Transport": transport,
    "Nearby_Schools": schools,
    "Nearby_Hospitals": hospitals,
    "Parking": parking,
    "Security": security,
    "Price_in_Lakhs": price_in_lakhs,
})

# ---------- Step 4: create the Good_Investment label ----------
# RULE (I decided this rule myself, it is not in any real dataset):
# A property is a "good investment" if its price per square foot is
# BELOW the median price per square foot of its own city.
# Meaning: it is cheap compared to other properties in the same city.
psf = data["Price_in_Lakhs"] * 100000 / data["Size_in_SqFt"]
city_median_psf = psf.groupby(data["City"]).transform("median")
data["Good_Investment"] = (psf < city_median_psf).astype(int)

# ---------- Step 5: add a little messiness ----------
# Real data always has missing values, so we add some here.
# This gives the cleaning step in train.py something real to do.
missing_rows = data.sample(60, random_state=1).index
data.loc[missing_rows, "Nearby_Schools"] = np.nan

missing_rows2 = data.sample(50, random_state=2).index
data.loc[missing_rows2, "Furnished_Status"] = np.nan

data.to_csv("data/property_data.csv", index=False)

print("Dataset created: data/property_data.csv")
print("Rows:", len(data), " Columns:", len(data.columns))
print("Good investments:", data["Good_Investment"].sum(), "out of", len(data))
print("Missing values added:", data.isnull().sum().sum())
