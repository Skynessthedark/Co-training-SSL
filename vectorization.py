#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 23:12:10 2019

@author: goksenin
"""

import numpy as np

class Vectorizer(object):
    
    def __init__(self, features):
        self.features = features
        
    def doc_vectorizer(self, doc):
        doc_vector = []
        for feature in self.features:
            count = 0
            for word in doc:
                if word == feature:
                    count += 1
            doc_vector.append(count)
        return doc_vector
