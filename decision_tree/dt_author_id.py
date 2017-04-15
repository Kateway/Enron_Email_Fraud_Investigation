#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
# Choose a smaller dataset
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]


clf = tree.DecisionTreeClassifier(min_samples_split = 40)
t0 = time()
clf = clf.fit(features_train, labels_train)
print "Training time:", round(time() - t0, 2), "s"

accurary = clf.score(features_test, labels_test)
t1 = time()
pred = clf.predict(features_test)
print "Predicting time:", round(time() - t1, 2), "s"

acc = accuracy_score(pred, labels_test)
print "Accuracy:", acc

#########################################################

print features_train.shape
