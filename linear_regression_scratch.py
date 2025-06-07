# 📘 Linear Regression from Scratch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Medical Price Dataset.csv")

# 📌 Data Preprocessing
df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
X = df.drop(columns=["charges"])
y = df["charges"]

# Normalize features
X = (X - X.mean()) / X.std()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)

# 🧮 Linear Regression Implementation (Gradient Descent)
def linear_regression(x_train, y_train, lr=0.01, epochs=1000):
    n_samples, n_features = x_train.shape
    x_train = np.c_[np.ones(n_samples), x_train]  # Add bias column
    
    weights = np.zeros(x_train.shape[1])

    for epoch in range(epochs):
        y_pred = x_train @ weights
        error = y_pred - y_train
        gradient = x_train.T @ error / n_samples
        weights -= lr * gradient
    
    return weights

# Train the model
weights = linear_regression(x_train, y_train)

# Evaluate on test set
x_test_bias = np.c_[np.ones(x_test.shape[0]), x_test]
y_pred = x_test_bias @ weights

# 📈 Visualization
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Linear Regression Prediction vs Actual")
plt.grid(True)
plt.show()

# 📊 Compute R^2 Score
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - y_pred) ** 2)
r2_score = 1 - (ss_res / ss_total)
print(f"R^2 Score on Test Data: {r2_score:.4f}")
