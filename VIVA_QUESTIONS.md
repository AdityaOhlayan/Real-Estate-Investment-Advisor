# Viva Questions and Answers

28 questions your examiner is likely to ask, with short answers that match
what the code actually does.

---

## Part A — The Project

**1. What is the objective of your project?**

To help a property buyer with two decisions. First, predict a fair price for
a property. Second, predict whether that property is a good investment,
meaning it is cheap compared to other properties in the same city.

---

**2. What is your dataset?**

A simulated dataset of 2000 properties with 13 columns, stored in
`data/property_data.csv`. I generated it myself using `make_dataset.py`.
It is not real market data, and I have stated that in my README.

---

**3. Why did you use a simulated dataset instead of a real one?**

I did not have access to a reliable real Indian property dataset with all
the columns I needed. Instead of pretending my data was real, I generated it
openly with a documented formula, so anyone can see exactly what is in it.

---

**4. If the data is simulated, is the machine learning meaningful?**

Yes, within limits. I built random noise into the price, so no model can be
perfect. The model has to genuinely learn the relationship between size,
city, age and price. But I am honest that it is learning a pattern I created,
not real market behaviour. That is listed as limitation number one.

---

**5. How many features and how many targets do you have?**

11 input features and 2 targets. `Price_in_Lakhs` is the regression target
and `Good_Investment` is the classification target. After `pd.get_dummies()`
the 11 features become 24 numeric columns.

---

## Part B — Data and Preprocessing

**6. What is a feature?**

A feature is an input column that the model uses to make its prediction.
In my project, `Size_in_SqFt`, `City` and `BHK` are features.

---

**7. What is the target variable?**

The target is the answer column that the model is trying to predict.
I have two: `Price_in_Lakhs` for regression and `Good_Investment` for
classification.

---

**8. Why did you clean the data?**

My dataset had 110 missing values. Scikit-learn models throw an error if
they see a missing value, so every row must be complete before training.

---

**9. How did you fill the missing values?**

`Nearby_Schools` is a number, so I filled it with the **median**, the middle
value. `Furnished_Status` is text, so I filled it with the **mode**, the most
common value. I used the median rather than the mean because the median is
not pulled around by extreme values.

---

**10. Why did you use `pd.get_dummies()`?**

Machine learning models can only do maths on numbers, and `City` contains
text like "Mumbai". `get_dummies()` turns one text column into several 0/1
columns, such as `City_Mumbai` and `City_Pune`. This is called one-hot
encoding.

---

**11. Why not just number the cities 1, 2, 3, 4, 5, 6?**

Because the model would then think Mumbai is "bigger than" Delhi, or that
city 6 is three times city 2. Those comparisons are meaningless for city
names. One-hot encoding avoids inventing a false order.

---

**12. What is scaling and why did you use it?**

Scaling puts all the columns on a similar range. In my data, price is in the
thousands while BHK is only 1 to 5, so price would dominate. I used
`StandardScaler` for the classification task because Logistic Regression is
sensitive to scale. Decision Trees do not need it, since they only compare
values.

---

**13. What is EDA?**

Exploratory Data Analysis. It means examining the data using tables and
charts **before** building a model, so I understand what I am working with.

---

**14. What did you learn from your EDA?**

Four things. Prices are skewed, with most properties cheap and a few very
expensive. Size and price rise together. Mumbai is far more expensive than
the other cities, so `City` matters. And `Nearby_Schools` and
`Nearby_Hospitals` have almost zero correlation with price, so they are
useless columns in this dataset.

---

## Part C — Training

**15. Why did you split the data?**

To test the model on data it has never seen. If I tested on the same rows I
trained on, the model could simply memorise them and get a perfect score
that means nothing.

---

**16. What is training data and what is testing data?**

Training data is the 80% the model learns from. Testing data is the held-back
20% used only to check the model afterwards. I used
`train_test_split(X, y, test_size=0.2, random_state=42)`, which gives 1600
training rows and 400 testing rows.

---

**17. What does `random_state=42` do?**

It fixes the random shuffling so the split is the same every time I run the
code. Without it, my scores would change slightly on every run and would not
be reproducible. 42 is not special, any number works.

---

**18. What is classification?**

Predicting a category or a label. My classification model predicts
`Good_Investment`, which is either 0 or 1. There is no in-between answer.

---

**19. What is regression?**

Predicting a continuous number. My regression model predicts
`Price_in_Lakhs`, which can be 45.2 or 132.7 or any value in that range.

---

**20. Why did you choose these particular algorithms?**

For regression, Linear Regression is the simplest possible model and gives a
baseline, and a Decision Tree Regressor can capture relationships that are
not straight lines. For classification, Logistic Regression is the standard
starting point for a yes/no problem, and a Decision Tree Classifier gives a
different style of model to compare it against. I used two per task rather
than ten, so I can actually explain each one.

---

**21. How does a Decision Tree work?**

It asks a series of yes/no questions about the features, like "is the size
above 1500 square feet?". Each answer sends the row down a branch. At the
end of the branch it predicts either the average price (for regression) or
the most common class (for classification).

---

**22. Which models won, and why?**

The Decision Tree Regressor won for price, with R² 0.893 against Linear
Regression's 0.876. The tree can capture the fact that each city has its own
price level, which a single straight line struggles with. Logistic Regression
won for classification with 74.3% accuracy against the tree's 63.2%.

---

## Part D — Evaluation

**23. What is accuracy?**

The percentage of predictions that were correct. My classification accuracy
is 74.3%, meaning about 297 of the 400 test properties were labelled
correctly.

---

**24. Why is your accuracy only 74% and not 99%?**

Because the label depends on price *per square foot*, and I never give the
model that column. It only sees price and size separately and has to work out
the relationship itself. A score of 74% is an honest result. If it were 99%,
it would mean the answer had leaked into the inputs.

---

**25. What is a confusion matrix?**

A table showing correct and incorrect predictions for each class. Mine is:

```
              Predicted 0   Predicted 1
Actual 0          162            52
Actual 1           51           135
```

162 and 135 are correct. 52 and 51 are the mistakes.

---

**26. What are MAE, RMSE and R²?**

- **MAE** (Mean Absolute Error) — the average size of the mistake. Mine is
  19.55, so the price prediction is off by about 19.55 lakhs on average.
- **RMSE** (Root Mean Squared Error) — similar, but squares the errors first,
  so large mistakes count more heavily. Mine is 29.96 lakhs.
- **R²** — how much of the variation in price the model explains, from 0 to 1.
  Mine is 0.893, so about 89%.

---

**27. Why use accuracy for one task and R² for the other?**

Because they are different problems. Accuracy counts right and wrong answers,
which only makes sense for categories. R² measures how close a predicted
number is to the true number, which only makes sense for continuous values.

---

**28. Is accuracy a fair measure for your data?**

Yes, here it is, because my label is balanced at almost exactly 50/50. If 95%
of properties were labelled 0, a model that always guessed 0 would score 95%
while being useless. I checked this balance in EDA question 6.

---

**29. What is overfitting?**

When a model memorises the training data instead of learning the general
pattern. It scores very well on training data but badly on new data. I
limited my Decision Trees to `max_depth=8` to prevent this — an unlimited
tree would keep splitting until it memorised every single row.

---

**30. What is underfitting?**

The opposite: the model is too simple to capture the pattern, so it does
badly on both training and testing data. My Linear Regression slightly
underfits, since one straight line cannot represent six cities with very
different price levels.

---

**31. What is data leakage, and did you avoid it?**

Data leakage is when information about the answer accidentally gets into the
inputs, giving a fake high score. I avoided it in the regression task by
dropping `Good_Investment` from the features, because that label was
calculated *from* the price. Leaving it in would have handed the model a
large hint about the answer.

---

## Part E — The Application

**32. Why did you use Streamlit?**

Because it turns a Python script into a web app with almost no extra code. I
did not have to learn HTML, CSS or JavaScript. It lets me demonstrate the
trained model interactively instead of just printing numbers in a terminal.

---

**33. What does joblib do in your project?**

It saves a trained model to a file. Without it, the Streamlit app would have
to retrain the models every time it started, which would be slow. `train.py`
saves them once and `app.py` loads them instantly.

---

**34. Why do you save the column names as well as the models?**

Because `get_dummies()` on a single row from the app only creates the columns
that row happens to need. If the user picks Pune, it makes `City_Pune` but not
`City_Mumbai`. The model expects every column, in the exact order it was
trained on — 24 for the price model and 25 for the investment model. I saved
both column lists and use `reindex()` to rebuild the row correctly, filling
the missing columns with 0.

---

**35. Is the 5-year value a machine learning prediction?**

No, and the app says so on screen. It is plain compound interest:
`future value = price × (1 + rate)^5`, where the user chooses the growth
rate. I did not model it, because my dataset has no historical prices to
learn growth from. Claiming it was a prediction would be dishonest.

---

**36. What are the limitations of your project?**

The dataset is simulated rather than real. `Good_Investment` is a rule I
invented, not a real financial definition. The 5-year value is assumed rather
than predicted. There are only 6 cities and 2000 rows. Two of my columns turn
out to be useless. And there is no time information, so market cycles are
invisible to the model.

---

**37. How would you improve this project?**

Use a real dataset with actual historical prices, so the 5-year value can be
genuinely predicted. Add rental yield so "good investment" reflects real
returns instead of my rule. Use cross-validation rather than a single split.
And test Random Forest as a third model.
