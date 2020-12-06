# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:20:59 2019

@author: Mahip
"""

from CCDPS_LogisticRegression import LRaccuracy
from CCDPS_KNN import KNNaccuracy
from CCDPS_SVM import SVM_Linear, SVM_Kernel
from CCDPS_NaiveBayes import NBaccuracy


print('The accuracy provided by KNN: ', KNNaccuracy())
print('The accuracy provided by Logistic Regression: ' , LRaccuracy())
print('The accuracy provided by Gaussian Naive Bayes: ', NBaccuracy())
print('The accuracy provided by Linear SVM: ', SVM_Linear())
print('The accuracy provided by Kernel SVM: ', SVM_Kernel())
