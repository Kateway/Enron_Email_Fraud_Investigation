#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import pprint

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

# pprint.pprint(data_dict)

def find_highest_salary():
	highest_salary = 0
	highest_salary_name = ""
	highest_salary_record = {}

	for record in data_dict:
		if data_dict[record]["salary"] == "NaN":
			continue
		elif data_dict[record]["salary"] > highest_salary:
			highest_salary = data_dict[record]["salary"]
			highest_salary_name = record
			highest_salary_record = data_dict[record]

	print highest_salary_name
	pprint.pprint(highest_salary_record)

# Find the highest salary and remove
print "Highest salary:"
find_highest_salary()
data_dict.pop("TOTAL", 0)

# Find the second highest salary and remove
print "Second highest salary:"
find_highest_salary()

# Third highest salary
data_dict.pop("SKILLING JEFFREY K", 0)
print "Third highest salary:"
find_highest_salary()

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

# Scatter plot of salary and bonus
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

