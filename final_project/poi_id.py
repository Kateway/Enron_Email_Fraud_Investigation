#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

import pprint
# print "Printing data dictionary keys:"
# pprint.pprint (data_dict.keys())

# Features list
financial_features = ['salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi']

poi_feature = ['poi']
features_list = poi_feature + email_features + financial_features
features_list_no_poi = email_features + financial_features

# print "Printing features list:"
# pprint.pprint(features_list)

# Numbers of poi and non-poi.
def print_number(data_dict):
	print "There are a total of", len(data_dict), "records at this point."
	number_of_poi = 0
	for record in data_dict.keys():
		number_of_poi += data_dict[record]['poi']
	print "There are", number_of_poi, "people of interest (poi) in the dataset at this point."
	number_of_none_poi =len(data_dict) - number_of_poi
	print "There are", number_of_none_poi, "people that are not poi in the dataset at this point."
	print '\n'

print "Printing numbers of poi and non-poi before any data cleaning:"
print_number(data_dict)

### Task 2: Remove outliers

##################################################################################################################################
######################################################## Remove outliers #########################################################
##################################################################################################################################
# Remove outlier found in the mini-project
data_dict.pop("TOTAL", 0)
print "Removed \"TOTAL\"."
print_number(data_dict)

# Look at the data dictionary keys, it appears that "THE TRAVEL AGENCY IN THE PARK" is not a person.  Remove it.
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
print "Removed \"THE TRAVEL AGENCY IN THE PARK\"."
print_number(data_dict)

# Person(s) with no financial features
nan_list = {}
for record in data_dict:
	nan_list[record] = 0
	for feature in features_list_no_poi:
		if data_dict[record][feature] == "NaN":
			nan_list[record] += 1
# sorted(nan_list.items(), key=lambda x:x[1])
# Sort nan_list by value
import operator
sorted_nan_list = sorted(nan_list.items(), key=operator.itemgetter(1))

# pprint.pprint(sorted_nan_list)
print "There are a total of", len(features_list_no_poi), "features except the poi label.", '\n'
# Key "LOCKHART EUGENE E" has NaN for all features. Remove it.
data_dict.pop("LOCKHART EUGENE E", 0)
print "Removed \"LOCKHART EUGENE E\"."
print_number(data_dict)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

##################################################################################################################################
#################################################### Create new features #########################################################
##################################################################################################################################

for record in my_dataset:
	message_from_poi = my_dataset[record]['from_poi_to_this_person']
	to_msg = my_dataset[record]['to_messages']
	message_to_poi = my_dataset[record]['from_this_person_to_poi']
	from_msg = my_dataset[record]['from_messages']
	if message_from_poi != "NaN" and to_msg != "NaN":
		my_dataset[record]['msg_ratio_from_poi'] = float(message_from_poi) / float(from_msg)
	else:
		my_dataset[record]['msg_ratio_from_poi'] = 0

	if message_to_poi != "NaN" and from_msg != "NaN":
		my_dataset[record]['msg_ratio_to_poi'] = float(message_to_poi) / float(to_msg)
	else:
		my_dataset[record]['msg_ratio_to_poi'] = 0

# print "Printing the dataset after adding new features:"
# pprint.pprint(my_dataset)
# Update the feature list
new_features_list = features_list + ['msg_ratio_to_poi', 'msg_ratio_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, new_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##################################################################################################################################
######################################## # Find k best features and save accuracy, precision, and recall #########################
##################################################################################################################################

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import preprocessing

# Algorithms
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.svm import SVC
from sklearn import tree

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import matplotlib.pyplot as plt

print "There are", len(new_features_list), "featrues in the new features list.", '\n'

algorithms = ["nb", "svc", "decision_tree"]
k_range = range(1, 22)
kbest_accuracies = []
kbest_precisions = []
kbest_recalls = []

def find_kbest(k, algorithm):
	k_best = SelectKBest(f_classif, k = k)
	kbest_features = k_best.fit_transform(features, labels)

	feature_scores = ['%.2f' % elem for elem in k_best.scores_ ]
	# print "Feature scores:"
	# print feature_scores

	feature_scores_pvalues = ['%.3f' % elem for elem in  k_best.pvalues_ ]
	# print "p values:"
	# print feature_scores_pvalues

	# Get the features score and sort it
	features_selected_tuple = [(new_features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in k_best.get_support(indices=True)]
	features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
	# print "Feature scores:"
	# pprint.pprint(features_selected_tuple)

	# Split the data
	features_train, features_test, labels_train, labels_test = \
	    train_test_split(kbest_features, labels, test_size=0.3, random_state=42)

	# Algorithms
	if algorithm == "nb":
		clf = GaussianNB()
	elif algorithm == "svc":
		clf = SVC(kernel = "rbf", C = 10000)
	elif algorithm == "decision_tree":
		clf = tree.DecisionTreeClassifier(min_samples_split = 10)

	clf = clf.fit(features_train, labels_train)
	kbest_accuracy = round(clf.score(features_test, labels_test), 3)
	pred = clf.predict(features_test)
	# print "Accuracy:", kbest_accuracy

	kbest_precision = round(precision_score(labels_test, pred), 3)
	# print "Precision score:", kbest_precision

	kbest_recall = round(recall_score(labels_test, pred), 3)
	# print "Recall:", kbest_recall

	# Save result to list
	kbest_accuracies.append(kbest_accuracy)
	kbest_precisions.append(kbest_precision)
	kbest_recalls.append(kbest_recall)

# Plot results
def plot_kbest_results(algorithm, kbest_accuracies, kbest_precisions, kbest_recalls):
	plt.scatter(k_range, kbest_accuracies, c = 'red', label = 'Accuracy')
	plt.scatter(k_range, kbest_precisions, c = 'blue', label = 'Precision')
	plt.scatter(k_range, kbest_recalls, c = 'orange', label = 'Recall')
	plt.legend();
	plt.xlabel("Number of k Best Features")
	plt.ylabel("Score")
	plt.title(algorithm)
	plt.show()

for k in k_range:
	for algorithm in algorithms:
		find_kbest(k, algorithm)

plot_kbest_results("Algorithm: Naive Bayes", kbest_accuracies[0:21], kbest_accuracies[0:21], kbest_precisions[0:21])
plot_kbest_results("Algorithm: SVC", kbest_accuracies[21:42], kbest_accuracies[21: 42], kbest_precisions[21:42])
plot_kbest_results("Algorithm: Decision Tree", kbest_accuracies[42:], kbest_accuracies[42:], kbest_precisions[42:])

# Best on the plots, the best algorithm of the three is naive bayes with three features.

##################################################################################################################################
########################################### Continue the analysis with three features ############################################
##################################################################################################################################

k_best = SelectKBest(f_classif, k = 3)
kbest_features = k_best.fit_transform(features, labels)
feature_scores = ['%.2f' % elem for elem in k_best.scores_ ]
feature_scores_pvalues = ['%.3f' % elem for elem in  k_best.pvalues_ ]

# Get the features score and sort it
features_selected_tuple = [(new_features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in k_best.get_support(indices=True)]
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
print '\n', "Three selected features and p values:"
pprint.pprint(features_selected_tuple)
print '\n'

# Split the data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(kbest_features, labels, test_size=0.3, random_state=42)

# Naive bayes
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
clf = GaussianNB(priors = None)
clf = clf.fit(features_train, labels_train)
# accurary = clf.score(features_test, labels_test)
# feature_scores_pvalues = ['%.3f' % elem for elem in  k_best.pvalues_ ]
accurary = clf.score(features_test, labels_test)
pred = clf.predict(features_test)

print "Accuracy:", round(accurary, 3)

from sklearn.metrics import precision_score
print "Precision score:", precision_score(labels_test, pred)

from sklearn.metrics import recall_score
print "Recall:", recall_score(labels_test, pred), '\n'

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

##################################################################################################################################
##################################################### Tune Parameters ############################################################
##################################################################################################################################
# The priors = None option was investigated in the previous section and there is no change in the results.


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# dump_classifier_and_data(clf, my_dataset, features_list)
dump_classifier_and_data(clf, my_dataset, new_features_list)

