# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:01:38 2024

@author: icon
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
com = pd.read_csv("C:/2-dataset/Company_Data.csv")
com.isnull().sum()
com.dropna()
com.columns

lb=LabelEncoder()
com["ShelveLoc"]=lb.fit_transform(com["ShelveLoc"])
com["Sales"]=lb.fit_transform(com["Sales"])
com["Urban"]=lb.fit_transform(com["Urban"])
com["US"]=lb.fit_transform(com["US"])


predictors=com.columns[:-1]
target='Sales'

#splitting data into trainning and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(com,test_size=0.3)

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



