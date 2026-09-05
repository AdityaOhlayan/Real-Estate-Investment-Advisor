"""
app.py
------
The Streamlit web app for the project. It has 3 pages:

  1. Home       - what the project is about
  2. Dataset    - a look at the data and two simple charts
  3. Prediction - enter a property and get the results

Run:  streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Real Estate Investment Advisor")

# ---------- Load the dataset and the trained models ----------
data = pd.read_csv("data/property_data.csv")

price_model = joblib.load("models/price_model.pkl")
investment_model = joblib.load("models/investment_model.pkl")
scaler = joblib.load("models/scaler.pkl")
price_columns = joblib.load("models/price_columns.pkl")
investment_columns = joblib.load("models/investment_columns.pkl")

# ---------- Sidebar menu ----------
page = st.sidebar.radio("Menu", ["Home", "Dataset", "Prediction"])


# =====================================================================
# PAGE 1: HOME
# =====================================================================
if page == "Home":
    st.title("Real Estate Investment Advisor")

    st.write("""
    This project helps a buyer look at a property and answer two questions:

    1. **What is a fair price for this property?**
       A regression model predicts the price in lakhs.

    2. **Is it a good investment?**
       A classification model predicts whether the property is cheap
       compared to other properties in the same city.

    The app also shows a simple estimate of the value after 5 years,
    using compound interest.
    """)

    st.warning(
        "This project uses a **simulated dataset** that I generated myself "
        "with `make_dataset.py`. It is for learning purposes and should not "
        "be used for real investment decisions."
    )

    st.subheader("How it was built")
    st.write("""
    - Loaded the dataset and filled the missing values
    - Converted text columns to numbers using `pd.get_dummies()`
    - Split the data into 80% training and 20% testing
    - Trained Linear Regression and a Decision Tree Regressor for the price
    - Trained Logistic Regression and a Decision Tree Classifier for the label
    - Kept the better model in each case and saved it with `joblib`
    """)


# =====================================================================
# PAGE 2: DATASET
# =====================================================================
elif page == "Dataset":
    st.title("The Dataset")

    st.write("Number of rows:", len(data))
    st.write("Number of columns:", len(data.columns))

    st.subheader("First 10 rows")
    st.dataframe(data.head(10))

    st.subheader("Basic statistics")
    st.dataframe(data.describe().round(2))

    st.subheader("Average price by city")
    average_price = data.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    st.bar_chart(average_price)

    st.subheader("Size vs Price")
    st.scatter_chart(data, x="Size_in_SqFt", y="Price_in_Lakhs")

    st.subheader("Good investment label")
    st.write(
        "A property is labelled a good investment (1) if its price per "
        "square foot is below the median for its own city."
    )
    st.bar_chart(data["Good_Investment"].value_counts().sort_index())


# =====================================================================
# PAGE 3: PREDICTION
# =====================================================================
else:
    st.title("Predict")
    st.write("Enter the property details below.")

    city = st.selectbox("City", sorted(data["City"].unique()))
    property_type = st.selectbox("Property Type", sorted(data["Property_Type"].unique()))
    furnished = st.selectbox("Furnished Status", ["Unfurnished", "Semi-Furnished",
                                                  "Fully-Furnished"])
    transport = st.selectbox("Public Transport", ["Low", "Medium", "High"])

    size = st.number_input("Size in SqFt", 400, 3500, 1200, step=50)
    bhk = st.number_input("BHK", 1, 5, 2)
    age = st.number_input("Age of property (years)", 0, 40, 5)

    schools = st.slider("Nearby schools", 0, 10, 5)
    hospitals = st.slider("Nearby hospitals", 0, 8, 4)

    parking = st.radio("Parking", ["Yes", "No"], horizontal=True)
    security = st.radio("Security", ["Yes", "No"], horizontal=True)

    asking_price = st.number_input("Asking price in lakhs", 5.0, 2000.0, 100.0, step=5.0)
    growth_rate = st.slider("Assumed yearly growth %", 1, 15, 7)

    if st.button("Predict"):
        # Put the user's answers into a one-row table, exactly like the
        # rows the model was trained on.
        new_property = pd.DataFrame([{
            "City": city,
            "Property_Type": property_type,
            "BHK": bhk,
            "Size_in_SqFt": size,
            "Age_of_Property": age,
            "Furnished_Status": furnished,
            "Public_Transport": transport,
            "Nearby_Schools": schools,
            "Nearby_Hospitals": hospitals,
            "Parking": parking,
            "Security": security,
            "Price_in_Lakhs": asking_price,
        }])

        # Convert the text columns to 0/1 columns, the same as in training.
        new_property_encoded = pd.get_dummies(new_property)

        # ---------- 1. Predict a fair price ----------
        # reindex puts the columns in the same order the model expects and
        # fills any column this single row does not have with 0.
        price_input = new_property_encoded.reindex(columns=price_columns, fill_value=0)
        predicted_price = price_model.predict(price_input)[0]

        # ---------- 2. Predict good investment or not ----------
        investment_input = new_property_encoded.reindex(
            columns=investment_columns, fill_value=0
        )
        investment_input_scaled = scaler.transform(investment_input)
        is_good_investment = investment_model.predict(investment_input_scaled)[0]

        # ---------- Show the results ----------
        st.subheader("Results")

        column1, column2 = st.columns(2)
        column1.metric("Predicted fair price", f"{predicted_price:,.1f} lakhs")
        column2.metric("Asking price", f"{asking_price:,.1f} lakhs")

        if predicted_price > asking_price:
            st.write("The asking price is **below** the predicted fair price.")
        else:
            st.write("The asking price is **above** the predicted fair price.")

        if is_good_investment == 1:
            st.success("Good investment: cheap for this city")
        else:
            st.info("Not a good investment: expensive for this city")

        # ---------- Value after 5 years ----------
        # This is NOT a machine learning prediction. It is a plain
        # compound interest calculation using a growth rate you choose.
        st.subheader("Value after 5 years")
        st.caption(
            "This is a simple compound interest calculation, not a "
            "machine learning prediction."
        )
        # Formula: future value = present value x (1 + rate) ^ years
        future_value = asking_price * (1 + growth_rate / 100) ** 5
        st.write(f"At {growth_rate}% growth per year:")
        st.write(f"**{future_value:,.1f} lakhs** after 5 years "
                 f"(a gain of {future_value - asking_price:,.1f} lakhs)")
