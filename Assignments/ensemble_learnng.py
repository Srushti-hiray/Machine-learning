# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:37:07 2024

@author: icon
"""

import pandas as pd
df = pd.read_csv("C:/2-dataset/diabetes_puma.csv")
df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()
#0    500
#1    268
#there is slight imbalance in our dataset but since it is not major
#we will not worry about it
#Train_test_split
X=df.drop("Outcome",axis="columns")
y=df.Outcome
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
X_scaled[:3]
#in order to make your data balanced while splitting, you can use stratify
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)
X_train.shape
X_test.shape
y_train.value_counts()
'''0    375
1    201'''
201/375
#0.536
y_test.value_counts()
'''0    125
1     67'''
67/125
#0.536

#train using strand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
#here k fold cross validation is used
scores=cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
scores
scores.mean()
#accuracy: 0.7097614803497156

#train using bagging
from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                              n_estimators=100,
                              max_samples=0.8,
                              oob_score=True,
                              random_state=0
                              )

bag_model.fit(X_train, y_train)
bag_model.oob_score_
# 0.7534722222222222
#note here we are not using test data, using OOB samples results are tested
bag_model.score(X_test, y_test)
#0.7760416666666666
#now let us apply cross validation
bag_model=BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
    )
scores=cross_val_score(bag_model, X, y, cv=5)
scores
scores.mean()
#0.7578728461081402

#we can see some improvements in test score with bagging classifier as compared to random forest 

#train using random forest
from sklearn.ensemble import RandomForestClassifier
scores=cross_val_score(RandomForestClassifier(n_estimators=50), X, y, cv=5)
scores.mean()




