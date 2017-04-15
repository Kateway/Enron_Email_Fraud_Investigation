#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

accurary = clf.score(features_test, labels_test)

pred = clf.predict(features_test)
print "Accuracy if splitting data into training and test datasets:", accurary

# Number of poi in the test set.
predicted_pois = 0
for prediction in pred:
	if prediction == 1:
		predicted_pois += 1

print "There are", predicted_pois, "pois predicted in the test set."
print "There are", len(labels_test), "people in the test set."

# Actual poi in test set
actual_poi = 0
for actual in zip(labels_test):
	if actual == 1:
		actual_poi += 1

print "There are", actual_poi, "actual poi in the test set."

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test, pred)
print "True negatives:", cm[0][0]
print "False negatives:", cm[1][0]
print "True positives:", cm[1][1]
print "False positives:", cm[0][1]

from sklearn.metrics import precision_score
print "Precision score:", precision_score(labels_test, pred)

from sklearn.metrics import recall_score
print "Recall:", recall_score(labels_test, pred)


