#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 00:13:59 2019

@author: goksenin
"""

import pandas as pd
import numpy as np
from Datasets import Dataset
from vectorization import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from cotraining import CotClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report



#Data paths
train_dir = "train/"
test_dir = "test/"
directory ="/home/goksenin/Desktop/GRADUATION PROJECT/Programming/"
dataset = Dataset(directory)
n_train = 3375
n_test = 1125
X_train, Y_train = dataset.get_set(train_dir)
X_test, Y_test = dataset.get_set(test_dir)

# -1: unlabeled 0:non-relative 1:relative 
y_train = np.asarray(Y_train)
y_train[n_train//4: ] = -1

#######FEATURE EXTRACTION
#getting related documents for feature_extraction
relative_index = [i for i, y_i in  enumerate(Y_train) if y_i == 1]
related_data = []
for index in relative_index:
    related_data.append(X_train[index])
for i, line in enumerate(related_data):
    space = " "
    related_data[i] = space.join(line)
#We will get 300 words which appears in only programming 
#documents and make them our features    
tf_idf = TfidfVectorizer(min_df=15, max_df=.85, lowercase=False,
                         max_features=300).fit(related_data)
vect_vocab = tf_idf.vocabulary_
vect_feat = tf_idf.transform(related_data)
features = tf_idf.get_feature_names()
feat_size = 300
# Vectorize each document with features
vectorizer = Vectorizer(features)
x_train_vect = [vectorizer.doc_vectorizer(i) for i in X_train]
x_test_vect = [vectorizer.doc_vectorizer(i) for i in X_test]
df_x_train = pd.DataFrame(x_train_vect, columns = features)
df_x_test = pd.DataFrame(x_test_vect, columns = features)
x1 = np.asarray(x_train_vect)[:, :-feat_size//2]
x2 = np.asarray(x_train_vect)[:, -feat_size//2:]
x1_test = np.asarray(x_test_vect)[:, :-feat_size//2]
x2_test = np.asarray(x_test_vect)[:, -feat_size//2:]

#Usual Naive Bayes
u_nb = GaussianNB()
u_nb.fit(df_x_train, Y_train)
y_p = u_nb.predict(df_x_test)
print('-----NB-------')
print(classification_report(Y_test, y_p))

# Co-training
nb1 = GaussianNB()
nb2 = GaussianNB()
clf = CotClassifier(nb1, nb2)
clf.fit(x1, x2, y_train)
y_proba = clf.predict_proba(x1_test, x2_test)
y_pred = clf.predict(x1_test, x2_test)
print('------COT--------')
print(classification_report(Y_test, y_pred))

