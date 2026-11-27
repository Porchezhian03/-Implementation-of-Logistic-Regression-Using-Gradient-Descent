# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the placement.csv dataset and select academic features (ssc_p, hsc_p, degree_p, etest_p, mba_p) and the target status.

2.Encode the target variable as Placed = 1 and Not Placed = 0, and standardize the input features.

3.Initialize the weights and bias, and define the Sigmoid function to calculate placement probability.

4.Calculate the prediction and gradient, then update the weights using Gradient Descent.

5.Repeat the gradient calculation and weight update for the specified number of iterations.

6.Predict the placement status using a 0.5 threshold and calculate the model accuracy.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PORCHEZHIAN E
RegisterNumber:212224040235
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv('/content/Placement_Data (2).csv')

# Drop unnecessary columns
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

# Convert categorical columns
cat_cols = ["gender", "ssc_b", "hsc_b", "degree_t",
            "workex", "specialisation", "status", "hsc_s"]

for col in cat_cols:
    data[col] = data[col].astype('category')

for col in cat_cols:
    data[col] = data[col].cat.codes

# Separate input and output
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Initialize theta
theta = np.random.randn(x.shape[1])

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    h = np.clip(h, 1e-10, 1 - 1e-10)
    return -np.sum(y * np.log(h) +
                   (1-y) * np.log(1-h))

# Gradient descent
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)

    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient

    return theta

# Train model
theta = gradient_descent(
    theta, x, y,
    alpha=0.01,
    num_iterations=1000
)

# Prediction function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Calculate accuracy
y_pred = predict(theta, x)
accuracy = np.mean(y_pred.flatten() == y)

# New student 1
xnew1 = np.array([[0, 87, 0, 95, 0, 2,
                   78, 2, 0, 0, 1, 0]])

y_prednew1 = predict(theta, xnew1)

# New student 2
xnew2 = np.array([[0, 0, 0, 0, 0, 2,
                   8, 2, 0, 0, 1, 0]])

y_prednew2 = predict(theta, xnew2)

# Display output
print("Accuracy: ", accuracy)
print("Prediction for Student 1:", y_prednew1)
print("Prediction for Student 2:", y_prednew2)

```

## Output:
<img width="1837" height="395" alt="image" src="https://github.com/user-attachments/assets/ef3d8ef2-8b16-43a1-9694-32ac035e778e" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

