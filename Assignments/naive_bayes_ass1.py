# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 15:40:30 2024

@author: icon
"""
'''
Bussiness objectives:
    1. Develop a classification model to predict 
   whether an individual's salary falls into one of the predefined 
   classes based on given features.'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
##Loading data
train_data=pd.read_csv("c:/2-dataset/SalaryData_Train.csv", encoding="ISO-8859-1")
test_data=pd.read_csv("c:/2-dataset/SalaryData_Test.csv", encoding="ISO-8859-1")

##Data Preprocessing
label_encoder = LabelEncoder()
train_data['maritalstatus'] = label_encoder.fit_transform(train_data['maritalstatus'])
test_data['maritalstatus'] = label_encoder.transform(test_data['maritalstatus'])
train_data['relationship'] = label_encoder.fit_transform(train_data['relationship'])
test_data['relationship'] = label_encoder.transform(test_data['relationship'])
train_data['race'] = label_encoder.fit_transform(train_data['race'])
test_data['race'] = label_encoder.transform(test_data['race'])
train_data['sex'] = label_encoder.fit_transform(train_data['sex'])
test_data['sex'] = label_encoder.transform(test_data['sex'])

##Split Data
X_train = train_data.drop('Salary', axis=1)
y_train = train_data['Salary']
X_test = test_data.drop('Salary', axis=1)
y_test = test_data['Salary']

##train the naive bayes
from sklearn.naive_bayes import MultinomialNB as MB
classifier_mb=MB()
classifier_mb.fit(train_data.Salary)


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
 
#####Loading data
email_train=pd.read_csv("c:/2-dataset/SalaryData_Train.csv",encoding="ISO-8859-1")
email_test=pd.read_csv("c:/2-dataset/SalaryData_Test.csv" ,encoding="ISO-8859-1")
 
from sklearn.model_selection import train_test_split
train_test_split(email_train,test_size=0.2)
 
########creating matrix of token counts for entire text documents####

emails_bow=CountVectorizer().fit(email_test.Salary) 
all_emails_matrix=emails_bow.transform(email_test.Salary)

####For training clients 

train_emails_matrix=emails_bow.transform(email_train.education)

###for testing clients
test_emails_matrix=emails_bow.transform(email_test.education)

#####Learning Term weightaging and normaling on entire clients salary
tfidf_transformer=TfidfTransformer().fit(all_emails_matrix)

######preparing TFIDF for train mails
train_tfidf=tfidf_transformer.transform(train_emails_matrix)

####preparing TFIDF for test mails
test_tfidf=tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape

######Now let us apply this to the Naive Bayes therorem

from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb=MB()
classifier_mb.fit(train_tfidf,email_train.workclass)

######Evalution on test data

test_pred_m= classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.workclass) 
accuracy_test_m

