# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:46:11 2024

@author: icon
"""
'''
Bussiness objectives:'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

zoo = pd.read_csv("C:/1-Dataset/Zoo.csv.xls")

# Identify non-numeric columns
non_numeric_columns = zoo.select_dtypes(exclude=['number']).columns

# Drop non-numeric columns
zoo = zoo.drop(columns=non_numeric_columns)

# Assuming the last column is the target variable ('diagnosis')
X = zoo.iloc[:, :-1].values
y = zoo.iloc[:, -1].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(pred, y_test))
print(pd.crosstab(pred, y_test))

# Select the correct value of k
acc = []

for i in range(3, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    acc.append([train_acc, test_acc])

# Plot the accuracy for different values of k
plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], "ro-", label='Training Accuracy')
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], "bo-", label='Testing Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Choosing the optimal value of k
optimal_k = 7
knn = KNeighborsClassifier(n_neighbors=optimal_k)
knn.fit(X_train, y_train)
pred_optimal_k = knn.predict(X_test)
print("Accuracy with optimal k:", accuracy_score(pred_optimal_k, y_test))
print(pd.crosstab(pred_optimal_k, y_test))