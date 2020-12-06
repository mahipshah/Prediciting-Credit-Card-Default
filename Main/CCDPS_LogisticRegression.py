# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 20:41:00 2019

@author: Mahip
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class LR:
    def __init__(self, lr=0.001, num_iters=1000):
        self.lr = lr
        self.num_iters = num_iters
        self.weights = None
        self.bias = None
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.num_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.__sigmoid(linear_model)
            
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.__sigmoid(linear_model)
        y_pred_classes = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_classes

def LRaccuracy():
    data = pd.read_csv("CreditCardDefault.csv")
    data = data.drop(data.columns[0], axis=1)

    norm_cols = ['bill_sept', 'bill_aug', 'bill_july', 'bill_june', 'bill_may',
             'bill_apr', 'paid_sept', 'paid_aug', 'paid_july', 'paid_june', 'paid_may', 
             'paid_apr']

    data[norm_cols] = data[norm_cols].apply(lambda x: (x - np.mean(x)) / np.std(x))

    X = data.iloc[:, 6:24]
    y = data.iloc[:, -1]
    
        
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    logreg = LR(lr=0.01, num_iters=500)
    logreg.fit(X_train, y_train)

    predictions = logreg.predict(X_test)

    conf_mat = confusion_matrix(y_test, predictions)

    
    return 100 * (conf_mat[0][0] + conf_mat[1][1]) / len(predictions)
