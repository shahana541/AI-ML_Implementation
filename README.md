# AI/ML Algorithm Implementations

Hands-on implementation of core ML algorithms, completed as part of AI/ML engineering learning tasks.

## Algorithms Implemented

| # | Algorithm | Type | Dataset | R² Score |

| 1 | Linear Regression | Regression | Salary vs Experience | 0.236 |
| 2 | Decision Tree | Regression | Salary (multi-feature) | 0.815 |
| 3 | Random Forest | Regression | Salary (multi-feature) | 0.834 |
| 4 | XGBoost | Regression | Salary (multi-feature) | 0.894 |
| 5 | LSTM | Time Series | Apple Stock Price | 0.9546 |

## Project Structure

## Key Learnings

- Built a complete regression pipeline: load → clean → encode → split → train → evaluate
- Handled missing values and inconsistent categorical data in real-world datasets
- Compared label encoding (ordinal data) vs one-hot encoding (nominal data)
- Explored ensemble methods (Random Forest, XGBoost) and how they improve on single Decision Trees
- Implemented an LSTM neural network for time-series forecasting, including sequence windowing and data scaling
- Version-controlled the entire project using Git and GitHub

## Tools & Libraries

Python, pandas, numpy, scikit-learn, XGBoost, TensorFlow/Keras, matplotlib