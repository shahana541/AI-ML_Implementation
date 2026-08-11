import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("data.csv")
print(df.head())
print(df.describe())
print(df['experience_level'].unique())
experience_map = {
    "Entry (0-2 yrs)" : 0,
    "Mid (3-5 yrs)" : 1,
    "Senior (6-9 yrs)" : 2,
    "Lead (10+ yrs)" : 3
}
df['experience_level_num'] = df['experience_level'].map(experience_map)
print(df[['experience_level', 'experience_level_num']].head())
X = df[['experience_level_num']]   # double brackets → keeps it as a DataFrame (2D)
y = df['annual_salary_usd']        # single brackets → a Series (1D)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)