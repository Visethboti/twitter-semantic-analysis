# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 11:51:44 2022
"""


import pandas as pd

import re
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def processText(text):
    # replace all char that is not a-z with space
    text = re.sub('[^a-zA-Z]', ' ', text)
    # tokenize and lowercase everything
    text = text.lower().split()
    l = ''
    for w in text:
        w = lemmatizer.lemmatize(w) # apply lemmatization
        if(w not in stop_words): # filter out stop_words
            l = l+' '+w
    return l

def processDataset(path):
    l = []
    # importing the dataset
    df = pd.read_csv(path)
    for column in list(df.columns):
        text = column
        text = processText(text)
        l.append(text)
    return l


# ts stores the time in seconds
import time
ts = time.time()

# Importing the dataset and preprocess it
negativeDataset = processDataset('../Dataset/processedNegative.csv')
positiveDataset = processDataset('../Dataset/processedPositive.csv')
neutralDataset = processDataset('../Dataset/processedNeutral.csv')




# Feature Extration/Vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(negativeDataset + positiveDataset + neutralDataset).toarray()
y = ['negative'] * len(negativeDataset) + ['positive'] * len(positiveDataset) + ['neutral'] * len(neutralDataset)

# split dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)

# Classifier 
from sklearn import svm
classifier = svm.SVC() #decision_function_shape='ovo'
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculate the accuracy
accuracy = (cm[0,0]+cm[1,1]+cm[2,2])/len(X_test)



df = pd.read_csv("report.csv")
time_took = time.time() - ts
df.loc[1,["classifier","time_took", "accuracy"]] = ["Support Vector Machine", time_took, accuracy]
df.to_csv("report.csv", index=False)