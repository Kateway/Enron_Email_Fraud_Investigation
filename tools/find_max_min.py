#!/usr/bin/python

import pickle
import sys
sys.path.append("../tools/")
import pprint


def find_max_min(data_dict, feature_name):
	max_value = 0
	max_value_name = ""
	min_value_name = ""
	max_value_record = {}
	min_value_record = {}

	for record in data_dict:
		# Find the maximum value.
		if data_dict[record][feature_name] == "NaN":
			continue
		elif data_dict[record][feature_name] > max_value:
			max_value = data_dict[record][feature_name]
			max_value_name = record
			max_value_record = data_dict[record]

		# Find the minimum level.
		min_value = max_value
		if data_dict[record][feature_name] == "NaN":
			continue
		elif data_dict[record][feature_name] < min_value:
			min_value = data_dict[record][feature_name]
			min_value_name = record
			min_value_record = data_dict[record]
	
	print "Name with highest value:"
	print max_value_name
	pprint.pprint(max_value_record)

	print "Name with the lowest value:"
	print min_value_name
	pprint.pprint(min_value_record)


