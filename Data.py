#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 14:34:26 2019

@author: goksenin
"""
import os
import pandas as pd
import numpy as np
from stemming import Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer


class Dataset(object):
    
    def __init__(self, directory):
        '''
        directory: dataset directory as string
        stemmer: Potter's Stemmer for stemming and tokenizing from nltk
        tf_vectorizer: object to vectorize datas which appears at least 15 documents for featuring
        '''
        self.directory = directory
        self.stemmer = Stemmer()
        
    def get_set(self, train_dir):
        X = []
        Y = []
        os.chdir(self.directory + train_dir)
        for root, dirs, files in os.walk('.'):
            for file in files:
                f = open(file, 'r', encoding='iso-8859-9')
                data = []
                for line in f:
                    if not line.startswith(("<ANCH>", "<P>")):
                        continue
                    else:
                        data += self.stemmer.stem_text(line)
                X.append(data)
                #if y is nonrelative ->0
                #else -> 1
                if file[0] == 'n':
                    Y.append(0)
                else:
                    Y.append(1)       
        return X, Y
 
        
    
    
    
