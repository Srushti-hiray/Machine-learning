# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:53:26 2024

@author: icon
"""

'''2. Divide the diabetes data into train and test datasets and build a 
Random Forest and Decision Tree model with Outcome as the output variable.'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
diab = pd.read_csv("C:/2-dataset/Diabetes.csv")
diab.isnull().sum()
diab.dropna()
diab.columns 

lb=LabelEncoder()
diab[" Class variable"]=lb.fit_transform(diab[" Class variable"])

diab[' Class variable'].unique
diab[' Class variable'].value_counts()
colnames=list(diab.columns)

predictors=colnames[:8]
target=colnames[8]

#splitting data into trainning and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(diab,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT
model=DT(criterion='entropy')
model.fit(train[predictors], train[target])
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target],preds_test,rownames=['Actual'], colnames=['predictions'])
np.mean(preds_test==test[target])

#now let us check accuracy on training dataset
#to check whether the model is overfit or underfit
preds_train=model.predict(train[predictors])
preds_train
pd.crosstab(train[target],preds_train,rownames=['Actual'], colnames=['predictions'])
np.mean(preds_train==train[target])
#accuracy on train data is greater than test data that is why this is overfit
#model