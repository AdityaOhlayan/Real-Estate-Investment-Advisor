# Real Estate Investment Advisor

A beginner-level Artificial Intelligence and Machine Learning mini project.

---

## 1. Project Title

**Real Estate Investment Advisor — Predicting Property Price and Investment Value**

---

## 2. Project Objective

When someone wants to buy a property, two questions matter most:

1. **What is a fair price for this property?**
2. **Is this property cheap or expensive compared to similar ones?**

This project answers both using machine learning:

| Task | Type | Target column |
|---|---|---|
| Predict the property price | Regression | `Price_in_Lakhs` |
| Predict if it is a good investment | Classification | `Good_Investment` |

The app also shows the value after 5 years. That part is **not** machine
learning — it is a plain compound interest calculation, and the app says so.

---

## 3. Dataset

The dataset is at `data/property_data.csv` and has **2000 rows and 13 columns**.

**This is a simulated dataset.** I generated it myself using
`make_dataset.py`. It is not real market data. I am stating this openly
because pretending it is real would be dishonest.

How the price was built inside the generator:

```
price per sqft = city base rate
                 x furnishing effect     (furnished costs more)
                 x age effect            (older costs less)
                 x transport effect      (good transport costs more)
                 x random noise          (nothing is perfectly predictable)

price in lakhs = price per sqft x size / 100000
```

The random noise is important. It means **no model can ever be 100% correct**,
which keeps the machine learning part honest.

### Columns

| Column | Meaning |
|---|---|
| `City` | One of 6 Indian cities |
| `Property_Type` | Apartment, Villa or Independent House |
| `BHK` | Number of bedrooms |
| `Size_in_SqFt` | Size in square feet |
| `Age_of_Property` | Age in years |
| `Furnished_Status` | Unfurnished, Semi-Furnished, Fully-Furnished |
| `Public_Transport` | Low, Medium, High |
| `Nearby_Schools` | Count of schools nearby |
| `Nearby_Hospitals` | Count of hospitals nearby |
| `Parking` | Yes or No |
| `Security` | Yes or No |
| `Price_in_Lakhs` | **Regression target** |
| `Good_Investment` | **Classification target** (0 or 1) |

### How `Good_Investment` is defined

This is a rule **I** wrote. It is not something found in real data:

> A property is a good investment (`1`) if its price per square foot is
> **below the median price per square foot of its own city**.

In plain words: it is cheap compared to other properties in the same city.
About 50% of the properties get the label `1`.

---

## 4. Technologies Used

| Tool | Why |
|---|---|
| Python | The programming language |
| Pandas | Loading and cleaning the data |
| NumPy | Generating the dataset |
| Matplotlib and Seaborn | Charts for the EDA |
| scikit-learn | The machine learning models |
| joblib | Saving trained models to files |
| Streamlit | The web app |

---

## 5. Data Preprocessing

All of this is in `train.py`, in order:

1. **Load the CSV** with `pd.read_csv()`
2. **Check missing values** with `data.isnull().sum()` — there are 110
3. **Fill missing values**
   - `Nearby_Schools` (a number) is filled with the **median**
   - `Furnished_Status` (text) is filled with the **mode**, the most common value
4. **Convert text to numbers** with `pd.get_dummies()`
   - `City` becomes `City_Mumbai`, `City_Delhi`, `City_Pune` and so on,
     each holding a 0 or a 1
   - This is needed because models can only do maths on numbers
5. **Separate X and y** — X is the features, y is the answer
6. **Split** into 80% training and 20% testing with `train_test_split()`
7. **Scale** the features for the classification task with `StandardScaler`,
   because price is in the thousands while BHK is only 1 to 5

There are no pipelines, no ColumnTransformer and no engineered score columns.
Every step is one readable line.

---

## 6. EDA (Exploratory Data Analysis)

EDA means looking at the data with tables and charts **before** building any
model. `eda.py` answers 6 questions and saves one chart for each into
`outputs/`.

| # | Question | Chart | What it tells us |
|---|---|---|---|
| 1 | How are prices spread out? | Histogram | Most properties are cheap, a few are very expensive |
| 2 | Does a bigger property cost more? | Scatter plot | Yes — correlation is about 0.43 |
| 3 | Which city is most expensive? | Bar chart | Mumbai is far ahead, so `City` matters a lot |
| 4 | Does furnishing change the price? | Box plot | Furnished properties cost a bit more |
| 5 | How are the numbers related? | Correlation heatmap | Size and BHK move together closely |
| 6 | Is the label balanced? | Bar chart | 50/50, so accuracy is a fair measure to use |

**One useful finding:** `Nearby_Schools` and `Nearby_Hospitals` have almost
zero correlation with price. They turned out to be useless columns. Finding a
useless feature is a real EDA result, not a failure.

---

## 7. Machine Learning Algorithms Used

Only four models, two per task.

### Regression — predicting `Price_in_Lakhs`
- **Linear Regression** — draws the best straight-line relationship between
  the features and the price
- **Decision Tree Regressor** — asks a series of yes/no questions
  ("is size above 1500?") and predicts the average price in each final group

`Good_Investment` is **removed** from the features here, because it was
calculated *from* the price. Leaving it in would be data leakage.

### Classification — predicting `Good_Investment`
- **Logistic Regression** — outputs a probability between 0 and 1 and rounds
  it to a class
- **Decision Tree Classifier** — same yes/no questions, but predicts a class

Here the price **is** allowed as a feature, because a buyer always knows the
asking price before deciding.

The script trains both models in each task, prints the scores, and keeps
whichever one scored better with a simple `if` statement.

---

## 8. Model Evaluation

Actual results from `python train.py`:

### Regression

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 24.39 lakhs | 32.21 lakhs | 0.876 |
| **Decision Tree Regressor** | **19.55 lakhs** | **29.96 lakhs** | **0.893** |

- **MAE** — on average the prediction is off by about 19.55 lakhs
- **RMSE** — similar, but punishes big mistakes more
- **R²** — 0.893 means the model explains about 89% of the variation in price

### Classification

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **0.743** |
| Decision Tree Classifier | 0.632 |

Confusion matrix for Logistic Regression:

```
              Predicted 0   Predicted 1
Actual 0          162            52
Actual 1           51           135
```

**Why is accuracy only 74% and not 99%?** Because the label depends on price
*per square foot*, and the model is never given that column. It only gets
price and size separately, and has to work the relationship out for itself.
A score of 74% is an honest result. If it were 99%, it would mean the answer
had been leaked into the inputs.

---

## 9. Streamlit Application

`app.py` has 3 pages, chosen from a sidebar menu:

1. **Home** — what the project does and how it was built
2. **Dataset** — first 10 rows, basic statistics, and two simple charts
3. **Prediction** — enter property details and press Predict

The Prediction page shows:
- the predicted fair price
- whether that is above or below the asking price
- whether the property is labelled a good investment
- the value after 5 years using compound interest, clearly marked as a
  calculation rather than a prediction

There is no database, no SQL, no filtering interface and no model management
screen. Just the three pages.

---

## 10. How to Run the Project

```bash
# 1. Install the libraries
pip install -r requirements.txt

# 2. Create the dataset (already included, but this regenerates it)
python make_dataset.py

# 3. Look at the data
python eda.py

# 4. Train the models
python train.py

# 5. Start the web app
streamlit run app.py
```

Run them in this order. Steps 2 and 4 must happen before step 5, because the
app loads the CSV and the saved model files.

### Files

```
project/
├── data/
│   └── property_data.csv       the dataset
├── models/                     saved models (created by train.py)
├── outputs/                    saved charts (created by eda.py)
├── make_dataset.py             creates the simulated dataset
├── eda.py                      6 EDA questions and charts
├── train.py                    trains and evaluates the 4 models
├── app.py                      the Streamlit web app
├── requirements.txt
├── README.md
└── VIVA_QUESTIONS.md
```

---

## 11. Limitations

I am listing these honestly rather than hiding them.

1. **The dataset is simulated.** I generated it with a formula. The model is
   learning patterns I put into the data, not real market behaviour.
2. **`Good_Investment` is a rule I invented.** A real "good investment"
   depends on rental yield, legal status, builder reputation, upcoming
   infrastructure and many other things this dataset does not have.
3. **The 5-year value is not predicted.** It is compound interest on a growth
   rate the user picks. I did not build a model for it, because the dataset
   has no historical prices to learn growth from.
4. **Only 6 cities and 2000 rows.** A real system would need lakhs of rows
   across many cities.
5. **`Nearby_Schools` and `Nearby_Hospitals` are useless here.** They do not
   affect price in this dataset, so the model ignores them.
6. **No time information.** Property markets rise and fall in cycles and this
   project cannot see that at all.

---

## 12. Future Improvements

1. Use a real dataset, for example a public housing dataset from Kaggle
2. Add real historical prices so the 5-year value can actually be predicted
   instead of assumed
3. Try Random Forest and compare it fairly against the current models
4. Use cross-validation instead of a single train/test split
5. Add rental yield, so "good investment" is based on real returns rather
   than on my rule
6. Deploy the app publicly on Streamlit Cloud
