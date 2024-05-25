# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:27:21 2024

@author: icon
"""
'''
 Bussiness objectives:
     1. Is to build a Naïve Bayes model to predict whether users in 
     the social network are likely to purchase the newly launched luxury SUV. 
     
     2. This is a binary classification problem, where 1 indicates a 
     purchase and 0 indicates no purchase.

'''
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MB

# Loading data
car_data = pd.read_csv("C:/Naive Baye's thm/NB_Car_Ad.csv.xls", encoding="ISO-8859-1")

# Splitting data into training and testing sets
car_train, car_test = train_test_split(car_data, test_size=0.2)

# Creating matrix of token counts for entire text document
car_bow = CountVectorizer().fit(car_data['Gender'])
all_car_matrix = car_bow.transform(car_data['Gender'])

# For training messages
train_car_matrix = car_bow.transform(car_train['Gender'])

# For testing messages
test_car_matrix = car_bow.transform(car_test['Gender'])

# Learning term weighting and normalizing for entire emails
tfidf_transformer = TfidfTransformer().fit(all_car_matrix)

# Preparing TFIDF for train models
train_tfidf = tfidf_transformer.transform(train_car_matrix)

# Preparing tfidf for test models
test_tfidf = tfidf_transformer.transform(test_car_matrix)

# Let's apply this to naive bayes theorem
classifier_mb = MB()
classifier_mb.fit(train_tfidf, car_train['Gender'])

# Evaluation on test data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == car_test['Gender'])
accuracy_test_m