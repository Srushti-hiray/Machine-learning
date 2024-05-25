# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:33:58 2024

@author: icon
"""
'''
Business Objective:
    Objective:
        Verify the authenticity of a candidate's salary claim based on 
        their stated experience 
        and previous monthly income.
        The accuracy of the model in predicting the monthly income of 
        candidates.

Constraints:
    
    1. Ensure that the data used for model training and testing complies with
    privacy regulations. 
    Salary information is sensitive, and handling it with care is essential.
    2. The model should aim for high accuracy to reliably predict candidates' monthly income. 
    Inaccurate predictions could lead to incorrect decisions in the 
    recruitment process.
          '''

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the recruitment dataset (assuming it's in a CSV file)
# Replace 'your_recruitment_dataset.csv' with the actual file name
recruitment_data = pd.read_csv("C:/2-dataset/HR_DT.csv")
recruitment_data.columns

# Display the first few rows of the dataset to understand its structure
recruitment_data.head()

# Encode categorical variables if needed (e.g., Position)
label_encoder = LabelEncoder()
recruitment_data['Position of the employee'] = label_encoder.fit_transform(recruitment_data['Position of the employee'])

# Separate features (X) and target variable (y)
X = recruitment_data.drop(' monthly income of employee', axis=1)  # Features
y = recruitment_data[' monthly income of employee']  # Target variable

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeRegressor

# Build a Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_decision_tree = decision_tree_model.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the Decision Tree model
mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
r2_decision_tree = r2_score(y_test, y_pred_decision_tree)
r2_decision_tree
#0.981709118123554
from sklearn.ensemble import RandomForestRegressor

# Build a Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_random_forest = random_forest_model.predict(X_test)

# Evaluate the Random Forest model
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
r2_random_forest = r2_score(y_test, y_pred_random_forest)
r2_random_forest
#0.9827462914905716
