import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("Salary_Data.csv")  # match exact filename
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['Job Title'].nunique())
df = df.dropna()
print(df.isnull().sum())
print(df.shape)
print(df['Gender'].unique())
print(df['Education Level'].unique())
df['Education Level'] = df['Education Level'].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
})
print(df['Education Level'].unique())
NewEducation_level = {
    "High School" : 0,
    "Bachelor's" : 1,
    "Master's" : 2,
    "PhD" : 3
}
df['Education_Level_num'] = df['Education Level'].map(NewEducation_level)
print(df['Education_Level_num'].isnull().sum())
df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
print(df.columns)
print(df.head())
X = df[['Age', 'Years of Experience', 'Education_Level_num', 'Gender_Male', 'Gender_Other']]
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)