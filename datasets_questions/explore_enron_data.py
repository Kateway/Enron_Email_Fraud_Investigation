#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# Number of records in the enron dataset.
print "There are", len(enron_data), "persons in the dataset."

# List of features
financial_features = ['salary', 'deferral_payments', 'total_payments', \
'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', \
'total_stock_value', 'expenses', 'exercised_stock_options', 'other', \
'long_term_incentive', 'restricted_stock', 'director_fees'] 

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', \
'from_this_person_to_poi', 'shared_receipt_with_poi', 'email_address']

poi_feature = ['poi']
features_list = poi_feature + email_features + financial_features
print "There are", len(features_list), "features of each person."

# Number of poi
number_of_poi = 0
for record in enron_data.keys():
	number_of_poi += enron_data[record]['poi']

print "There are", number_of_poi, "poi in the dtaset."

# poi names
poi_names = [
"(y) Lay, Kenneth",
"(y) Skilling, Jeffrey",
"(n) Howard, Kevin",
"(n) Krautz, Michael",
"(n) Yeager, Scott",
"(n) Hirko, Joseph",
"(n) Shelby, Rex",
"(n) Bermingham, David",
"(n) Darby, Giles",
"(n) Mulgrew, Gary",
"(n) Bayley, Daniel",
"(n) Brown, James",
"(n) Furst, Robert",
"(n) Fuhs, William",
"(n) Causey, Richard",
"(n) Calger, Christopher",
"(n) DeSpain, Timothy",
"(n) Hannon, Kevin",
"(n) Koenig, Mark",
"(y) Forney, John",
"(n) Rice, Kenneth",
"(n) Rieker, Paula",
"(n) Fastow, Lea",
"(n) Fastow, Andrew",
"(y) Delainey, David",
"(n) Glisan, Ben",
"(n) Richter, Jeffrey",
"(n) Lawyer, Larry",
"(n) Belden, Timothy",
"(n) Kopper, Michael",
"(n) Duncan, David",
"(n) Bowen, Raymond",
"(n) Colwell, Wesley",
"(n) Boyle, Dan",
"(n) Loehr, Christopher"
]

print "There are", len(poi_names), "poi names."

print enron_data

print "List of persons' names in the dataset:"
print enron_data.keys()

print "List of Features:"
print features_list

# Total value of stocks that belongs to James Prentice
james_prentice_stocks = enron_data["PRENTICE JAMES"]["total_stock_value"]
print "The total value of James Prentice is", james_prentice_stocks

# Number of email messages from Wesley Colwell to persons of interest
from_wesley_colwell_to_poi = enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "There are", from_wesley_colwell_to_poi, "messages from Wesley Colwell to poi."

# Value of stock options exercised by Jeffrey K Skilling
jeffrey_skilling_stock_options_value = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "The value of stock options of Jeffrey K Skilling is", jeffrey_skilling_stock_options_value, "."

# The enron CEO was Jeffrey Skilling.
# Kenneth Lay was the chair of Enron's board of directors.
# Andrew Fastow was the CFO
print "Jeffrey Skilling took home", enron_data["SKILLING JEFFREY K"]["total_payments"], "dollars (total payments)."
print "Kenneth Jay took home", enron_data["LAY KENNETH L"]["total_payments"], "dollars."
print "Andrew Fastow took home", enron_data["FASTOW ANDREW S"]["total_payments"], "dollars."

# Records with a value of a feature
def count_valid_value(feature_name):
	number_of_nan = len(enron_data)
	for record in enron_data.keys():
		if enron_data[record][feature_name] == "NaN":
			number_of_nan -= 1
	return number_of_nan

# Number of Nan of a feature for poi only
def count_poi_nan(feature_name):
	number_of_poi_nan = 0
	for record in enron_data.keys():
		if (enron_data[record][feature_name] == "NaN") and (enron_data[record]["poi"]):
			number_of_poi_nan += 1
	return number_of_poi_nan

print "There are", count_valid_value("salary"), "people with qualified salary."
print "There are", count_valid_value("email_address"), "people with known email address."
print len(enron_data) - count_valid_value("total_payments"), "people in the dataset has \"NaN\" for their total payments."
print "There are", float(1 - float(count_valid_value("total_payments")) / float(len(enron_data))), "with \"NaN\" as total payments."

print "There are", float(count_poi_nan("total_payments")) / float(number_of_poi), "poi with missing total payments."


