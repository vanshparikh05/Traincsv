
# Linear Regression from Scratch

This project demonstrates a simple implementation of **Linear Regression** using only NumPy, written from scratch without using any machine learning libraries like Scikit-learn.

## 📁 File Overview

- `linear_regression_scratch.ipynb`: Jupyter Notebook containing the full code and training process.

## 📌 Key Concepts

### 🔧 Imports
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
```

The script starts by importing necessary libraries: NumPy for numerical computation and Matplotlib for visualization.

### 📊 Data Generation
```python
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

```
Synthetic linear data is generated using NumPy, adding some noise to simulate real-world data.

### 🧠 Model Implementation
```python
 ...
```
A simple Linear Regression model is implemented using a Python class. It includes methods for prediction, loss calculation (Mean Squared Error), and gradient computation for updating weights.

### 🔁 Training Loop
```python
    for epoch in range(epochs):
        y_pred = x_train @ weights
        error = y_pred - y_train
        gradient = x_train.T @ error / n_samples
        weights -= lr * gradient
     ...
```
The model is trained using batch gradient descent. During each epoch, gradients are calculated and used to update the model parameters.

### 📈 Prediction Plot
```python
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') ...
```
After training, the predicted line is plotted alongside the original data points to visualize the performance of the trained model.

---

## ✅ Usage

You can run the notebook using:

```bash
jupyter notebook linear_regression_scratch.ipynb
```

Make sure you have `numpy` and `matplotlib` installed:

```bash
pip install numpy matplotlib
```

---

## 📬 Author

Made by Vansh Parikh.
