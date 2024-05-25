# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:05:07 2024

@author: icon
"""

'''
  Bussiness understanding:
      The dataset appears to be related to diabetes, suggesting a potential 
      application in the healthcare domain. 
      Understanding and predicting diabetes can be crucial for early 
      intervention and effective management of the disease.
      
  Bussiness objectives:
      The main objective is likely to develop predictive models using 
      machine learning techniques to predict the likelihood of diabetes 
      based on the given features.
      
      '''

import pandas as pd

diabetes_data = pd.read_csv("C:/2-dataset/Diabetes.csv")

# Display the first few rows of the dataset to understand its structure
diabetes_data.head()

X = diabetes_data.drop('Class variable', axis=1)  # Features
y = diabetes_data['Class variable']  # Target variable

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier

# Build a Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_decision_tree = decision_tree_model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

# Evaluate the Decision Tree model
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
accuracy_decision_tree
classification_report(y_test, y_pred_decision_tree)
#0.7467532467532467
from sklearn.ensemble import RandomForestClassifier

# Build a Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_random_forest = random_forest_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)
accuracy_random_forest
classification_report(y_test, y_pred_random_forest)
#0.7207792207792207
