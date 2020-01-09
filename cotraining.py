#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:12:55 2020

@author: goksenin
"""
import random
import numpy as np

class CotClassifier(object):
    
    def __init__(self, classifier1, classifier2, iter=45, u_=100):
        self.clf1 = classifier1
        self.clf2 = classifier2
        self.iter = iter
        self.u_ = u_
        random.seed()
        
    def fit(self, x1, x2, y, r_=1, n_=3):
        
        y = np.asarray(y)
        
        U = [i for i, y_i in enumerate(y) if y_i == -1]
        L = [i for i, y_i in enumerate(y) if y_i != -1]
        
        random.shuffle(U)
        U_ = U[-min(len(U), self.u_):]
        U = U[:-len(U_)]
        
        iter = 0
        while iter < self.iter and U:
            iter += 1
            self.clf1.fit(x1[L], y[L])
            self.clf2.fit(x2[L], y[L])
            
            y1_prob = self.clf1.predict_proba(x1[U_])
            y2_prob = self.clf2.predict_proba(x2[U_])
            
            r, n = [], []
            
            for i in (y1_prob[:, 0].argsort())[-r_:]:
                if y1_prob[i, 0] > 0.5:
                    r.append(i)
            for i in (y1_prob[:, 0].argsort())[-n_:]:
                if y1_prob[i, 0] > 0.5:
                    n.append(i)
            
            for i in (y2_prob[:, 0].argsort())[-r_:]:
                if y2_prob[i, 0] > 0.5:
                    r.append(i)
            for i in (y2_prob[:, 0].argsort())[-n_:]:
                if y2_prob[i, 0] > 0.5:
                    n.append(i)
            
            y[[U_[i] for i in r]] = 1
            y[[U_[i] for i in n]] = 0
            
            L.extend([U_[i] for i in r])
            L.extend([U_[i] for i in n])
            
            U_ = [x for x in U_ if not (x in r or x in n)]
            
            num = 0
            labeled_num = len(r) + len(n)
            while num != labeled_num and U:
                num += 1
                U_.append(U.pop())
        
        self.clf1.fit(x1[L], y[L])
        self.clf2.fit(x2[L], y[L])
        
    def predict_proba(self, x1, x2):
        y_prob = np.full((x2.shape[0], 2), -1)
        
        y1_prob = self.clf1.predict_proba(x1)
        y2_prob = self.clf2.predict_proba(x2)
        
        for i, (y1_i, y2_i) in enumerate(zip(y1_prob, y2_prob)):
            y_prob[i][0] = (y1_i[0] + y2_i[0]) / 2
            y_prob[i][1] = (y1_i[1] + y2_i[0]) / 2    
        e = 0.0001
        #assert all(abs(sum(y) - 1) <= e for y in y_prob)
        return y_prob
    
    def predict(self, x1, x2):
        
        y1 = self.clf1.predict(x1)
        y2 = self.clf2.predict(x2)
        
        y_pred = np.asarray([-1] * x1.shape[0])
        
        for i, (y1_i, y2_i) in enumerate(zip(y1, y2)):
            if y1_i == y2_i:
                y_pred[i] = y1_i
            else:
                y_prob = self.predict_proba(x1, x2)              
                #if the prob of relativity is less then .5 then y_pred = 0 otherwise 1 
                if y_prob[i][0] == 1:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
        
        assert not (-1 in y_pred)
        return y_pred
            
            
            