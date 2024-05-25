# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:47:54 2024

@author: icon
"""


'''Business Objectives:
    Detect instances of fraud to minimize financial losses for the company.
    Streamline fraud detection processes to reduce manual efforts and improve efficiency.
    Provide a secure environment for customers, ensuring their trust in the business.


Constraint: 
    If real-time processing is essential, the model must be efficient enough for quick predictions.
    Ensure that the models are interpretable, especially in industries where 
    regulatory compliance and transparency are crucial.
    Cost of False Positives and False Negatives:
    Consider the financial implications of false positives (unnecessary investigations)
    and false negatives (missed fraudulent activities).
    Data Privacy and Security:
    Adhere to data privacy regulations and ensure that sensitive 
    information is handled securely.
    The model should be scalable to handle an increasing volume of 
    transactions as the business grows.
 '''

import pandas as pd

# Load your fraud data into a Pandas DataFrame (assuming 'fraud_data' is your DataFrame)
# Replace 'your_data.csv' with your actual data file
fraud_data = pd.read_csv("C:/Decision Tree/Fraud_check.csv.xls")

# Data preprocessing: Discretize taxable_income
fraud_data['risk'] = fraud_data['Taxable.Income'].apply(lambda x: 'Risky' if x <= 30000 else 'Good')

# Split the data into features (X) and target variable (y)
X = fraud_data.drop(['Taxable.Income', 'risk'], axis=1)  # Assuming other columns are features
y = fraud_data['risk']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predictions on the test set
dt_predictions = dt_model.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate Decision Tree model
print("Decision Tree Model:")
print("Accuracy:", accuracy_score(y_test, dt_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print("Classification Report:\n", classification_report(y_test, dt_predictions))

from sklearn.ensemble import RandomForestClassifier

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions on the test set
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest model
print("\nRandom Forest Model:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
print("Classification Report:\n", classification_report(y_test, rf_predictions))