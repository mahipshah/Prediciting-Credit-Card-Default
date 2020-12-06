
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:16:54 2019

@author: devanshsheth
"""

# Example of summarizing a dataset
from math import sqrt
from math import pi
from math import exp
from random import seed
from random import randrange
import pandas as pd


def NBaccuracy():
# Calculate the Gaussian probability distribution function for x
    def calculate_probability(x, mean, stdev):
    	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    	return (1 / (sqrt(2 * pi) * stdev)) * exponent
    
    # Split the dataset by class values, returns a dictionary
    def separate_by_class(dataset):
        
        separated = {}
        
        yes = dataset.loc[dataset[6] == 1]
        no = dataset.loc[dataset[6] == 0]
    	
        yesDb = summarize_dataset(yes)
        noDb = summarize_dataset(no)
        
        separated[1]=yesDb
        separated[0]=noDb
        
        return separated
    
    # Calculate the mean of a list of numbers
    def mean(numbers):
    	return sum(numbers)/float(len(numbers))
    
    # Calculate the standard deviation of a list of numbers
    def stdev(numbers):
    	avg = mean(numbers)
    	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    	return sqrt(variance)
    
    # Calculate the mean, stdev and count for each column in a dataset
    def summarize_dataset(dataset):
    	summaries = [(mean(dataset[column]), stdev(dataset[column]), len(dataset[column])) for column in dataset.columns]
    	del(summaries[-1])
    	return summaries
    
    # Split dataset by class then calculate statistics for each row
    def summarize_by_class(dataset):
        summaries = separate_by_class(dataset)
        return summaries
    
    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(summaries, row):
    	total_rows = sum([summaries[label][0][2] for label in summaries])
    	probabilities = dict()
    	for class_value, class_summaries in summaries.items():
    		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
    		for i in range(len(class_summaries)):
    			mean, stdev, _ = class_summaries[i]
    			probabilities[class_value] *= calculate_probability(row.iloc[i], mean, stdev)
    	return probabilities
    
    # Predict the class for a given row
    def predict(summaries, row):
    	probabilities = calculate_class_probabilities(summaries, row)
    	best_label, best_prob = None, -1
    	for class_value, probability in probabilities.items():
    		if best_label is None or probability > best_prob:
    			best_prob = probability
    			best_label = class_value
    	return best_label
    
     
    # Split a dataset into k folds
    def cross_validation_split(dataset, n_folds):
        dataset_split = list()
        dataset_copy1 = dataset
        dataset_copy = dataset_copy1.values.tolist()
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split    
    
    # Calculate accuracy percentage
    def accuracy_metric(actual, predicted):
    	correct = 0
    	for i in range(len(actual)):
    		if actual[i] == predicted[i]:
    			correct += 1
    	return correct / float(len(actual)) * 100.0
    
    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(dataset, algorithm, n_folds, *args):
        folds = cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            train_set = pd.DataFrame(train_set)
            test_set = pd.DataFrame(test_set)
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores
    
    # Naive Bayes Algorithm
    def naive_bayes(train, test):
        summarize = summarize_by_class(train)
        predictions = list()
        for i in range(len(test)):
            output = predict(summarize, test.iloc[i])
            predictions.append(output)
        return(predictions)
    
    # Test summarizing a dataset
    dataset = pd.read_csv('CreditCardDefault.csv')
    dataset = dataset.drop(dataset.columns[0], axis=1)
    dataset = dataset.drop(columns = ['id','gender', 'limit_bal','marital_status','education','age', 'bill_sept', 'bill_aug', 'bill_july', 'bill_june', 'bill_may', 'bill_apr', 'paid_sept', 'paid_aug', 'paid_july', 'paid_june', 'paid_may', 'paid_apr'])
    
    seed(1)
    # evaluate algorithm
    n_folds = 3
    scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
    return sum(scores)/float(len(scores))
