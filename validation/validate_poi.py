#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### Decision tree
from sklearn import tree
from time import time
from sklearn.metrics import accuracy_score

# Train all the data
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

accurary = clf.score(features, labels)

pred = clf.predict(features)
print "Accuracy if train all the data:", accurary


# Split data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)

# from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

accurary = clf.score(features_test, labels_test)

pred = clf.predict(features_test)
print "Accuracy if splitting data into training and test datasets:", accurary
