# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:30:56 2024

@author: icon
"""

'''
  Business Understanding:
      Objective: The primary goal is to identify the key attributes or factors that contribute to high sales in a cloth manufacturing company.
      Target Variable: Sales - This is the variable we are trying to predict.
      Features/Attributes: These are the variables that we believe may influence sales. They could include factors like product type, price, promotion, seasonality, customer demographics, etc.

 Constraints:
    Data Availability: Ensure that the dataset contains relevant information about sales and associated attributes/features.
    Model Interpretability: Decision trees and random forests are relatively interpretable models, which aligns well with the business's need to understand the factors driving sales.
    Resource Constraints: Consider computational resources and time constraints for model training and evaluation.
    Accuracy Requirement: While accuracy is important, interpretability may take precedence over model complexity.
    Data Quality: Ensure the quality and cleanliness of the dataset to avoid biases or misleading results.

Business Objective:
    The business objective of a cloth manufacturing company is to
    identify the various attributes that contribute to high sales. 
    To achieve this objective, we will build a decision tree and 
    random forest model with Sales as the target variable. Before 
    building the models, we'll first convert the Sales variable into 
    a categorical variable.

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('C:/2-dataset/Company_Data.csv')

# Assuming 'Sales' is one of the columns in your dataset
# Convert Sales into a categorical variable
# You can use any criteria to categorize sales, for example, low, medium, high
data['Sales_Category'] = pd.cut(data['Sales'], bins=3, labels=['Low', 'Medium', 'High'])

# Define features and target variable
X = data.drop(['Sales', 'Sales_Category'], axis=1)
y = data['Sales_Category']

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

# Build Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
from sklearn.ensemble import RandomForestClassifier

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluate models
dt_accuracy = accuracy_score(y_test, dt_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)

# Visualize confusion matrix
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

plt.tight_layout()
plt.show()