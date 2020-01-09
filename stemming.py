#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:04:48 2019

@author: goksenin
"""
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#nltk.download()
class Stemmer(object):
    
    def stem_text(self, text):
        potter = PorterStemmer()
        punctuations = "?:!.,;I<>"
        tags = ["anch", "p", "P"]
        stop_words = stopwords.words('english')
        token_words = word_tokenize(text)
        stemmed_words = []
        for word in token_words:
            if word in punctuations or word in stop_words or word.isnumeric() or word in tags:
                continue
            stemmed_words.append(potter.stem(word))
        return stemmed_words

        
        

