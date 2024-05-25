# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:50:28 2024

@author: icon
"""

'''
 Bussiness Objectives: 
1.The primary goal is to develop a Naïve Bayes model that can accurately 
predict whether a given tweet about a disaster is real or fake.

2.A binary classification problem where 1 indicates a real tweet about a 
disaster, and 0 indicates a fake tweet.

3.Choose relevant features: Identify key features in the tweet text that 
might indicate whether it's about a real disaster or not'''

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score

# Loading the Twitter data
twitter_data = pd.read_csv('C:/2-dataset/Disaster_tweets_NB.csv')

# Dataset has two columns text and target
X = twitter_data['text']
y = twitter_data['target']

# Handling Missing Data if any
twitter_data.dropna(inplace=True)

# Text Preprocessing
# Here we are trying to remove any special characters, URLs,etc.
import re
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+|\@\w+|\#", "", text, flags=re.MULTILINE)
    return text

twitter_data['text'] = twitter_data['text'].apply(preprocess_text)

#------------------------------------------------------------------

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Converting the tweet text into a bag-of-words representation using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')  
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

#------------------------------------------------------------------

from sklearn.naive_bayes import MultinomialNB 

# Initialize the Naive Bayes model
naive_bayes_model = MultinomialNB()

# Training the model
naive_bayes_model.fit(X_train_vectorized, y_train)

# Making predictions on the test set
y_pred = naive_bayes_model.predict(X_test_vectorized)

# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
# Display the results
accuracy