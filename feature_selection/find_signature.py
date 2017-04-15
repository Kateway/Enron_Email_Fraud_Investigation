#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
from time import time
import numpy as np


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
# max_df was 0.5, but it raised problems.

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here
print "Number of training data in the training data is", len(features_train)

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split = 40)
t0 = time()
clf = clf.fit(features_train, labels_train)
print "Training time:", round(time() - t0, 2), "s"

accurary = clf.score(features_test, labels_test)
t1 = time()
pred = clf.predict(features_test)
print "Predicting time:", round(time() - t1, 2), "s"

# Accuracy score
print "Score for training data:", clf.score(features_train, labels_train)
print "Score for test data:", clf.score(features_test, labels_test)

# Print feature importance
print "Printing feature importances."
feature_importances = clf.feature_importances_
most_important_feature = np.argmax(feature_importances)
print "The most important feture is", most_important_feature
print "The significance of the most important feature is", np.max(feature_importances)

vocabulary_list = vectorizer.get_feature_names()
print "The word that has the most important feature is", vocabulary_list[most_important_feature]


