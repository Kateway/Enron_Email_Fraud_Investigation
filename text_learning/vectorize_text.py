#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
from pprint import pprint

"""
# Get stopper words from NLTK
# download nltk first.
# import nltk
# nltk.download()
from nltk.corpus import stopwords
sw = stopwords.words("english")
print "There are", len(sw), "stopper words."
print "Printing stopper words:"

# Stemmer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print "The stemmer of responsiveness is", stemmer.stem("responsiveness")
print "The stemmer of responsivity is", stemmer.stem("responsivity")
print "The stemmer of responsive is", stemmer.stem("responsive")

# Bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
string1 = "hi Katie the self driving car will be late Best Sebastian"
string2 = "Hi Sebastian the machine learning class will be great great great Best Katie"
string3 = "Hi Katie the machine learning class will be most excellent"
email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit(email_list)
bag_of_words = vectorizer.transform(email_list)
# print bag_of_words
"""

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0

for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            print path
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            text = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            text = text.replace("sara", "")
            text = text.replace("shackleton", "")
            text = text.replace("chris", "")
            text = text.replace("germani", "")
            # Remove outliers in feature selection lesson
            text = text.replace("sshacklensf", "")
            text = text.replace("cgermannsf", "")


            ### append the text to word_data
            word_data.append(text)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            else:
                from_data.append(1)


            email.close()

print "emails processed"
from_sara.close()
from_chris.close()

print "word_data[152]:", word_data[152]

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )



### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = "english")
vectorizer.fit(word_data)
vocab_list = vectorizer.get_feature_names()
print "There are", len(vocab_list), "words in the vocabulary list."
print "Word number 34597 in the vocabulary list is", vocab_list[34597], "."


