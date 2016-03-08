import re
import random
import csv
import pandas as pd
import numpy as np
import pickle
import nltk
# from math import log
# from math import sqrt
# from nltk.classify import NaiveBayesClassifier
# from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn import metrics
# from sklearn.naive_bayes import MultinomialNB
# from sklearn import svm
# from sklearn.metrics import accuracy_score

def removePunctuation(text):
    """Removes punctuation, changes to lower case, and strips leading and trailing spaces.

    Note:
        Only spaces, letters, and numbers should be retained.  Other characters should should be
        eliminated (e.g. it's becomes its).  Leading and trailing spaces should be removed after
        punctuation is removed.

    Args:
        text (str): A string.

    Returns:
        str: The cleaned up string.
    """
    text = text.strip()
    punctuation_less_str = re.sub(r'[^A-Za-z0-9\s]|^\s|\s$',"",text)
    return punctuation_less_str.lower()

data = pd.read_json('dump.json')
train_text = data['title'] 
train_text = train_text.map(lambda x : removePunctuation(x))


# targets = data['category']
# targets = np.array(targets,dtype=str) # Need to convert into numpy array for Cross Validation funct of Sklearn to work

# print "Training on ",len(data), "instances using only title\n",


# clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1) )),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=100, random_state=50)) ,])
# clf = clf.fit(train_text.astype(str),targets)

# num_cross = 10
# scores = cross_validation.cross_val_score(clf, train_text, targets, cv=5)

# print "Cross validation on", num_cross,"splits"
# print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
# # -----------------------------------------------------------------------------------------------------------

# print "\nTraining on ",len(data), "instances using only brief\n",

# train_text = data['brief'] 
# train_text = train_text.map(lambda x : removePunctuation(x))
# train_text = pd.Series(train_text, dtype=str)


# clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,1) )),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=100, random_state=50)) ,])
# clf = clf.fit(train_text.astype(str),targets)

# num_cross = 10
# scores = cross_validation.cross_val_score(clf, train_text, targets, cv=5)

# print "Cross validation on", num_cross,"splits"
# print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

# # --------------------------------------------------------------------------------------------------------

# print "\nTraining on ",len(data), "instances using only title + brief\n",

# train_text = pd.Series(np.array(data["title"])+" "+np.array(data["brief"]))
# train_text = train_text.map(lambda x : removePunctuation(x))
# train_text = pd.Series(train_text, dtype=str)

# clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2) )),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=100, random_state=50)) ,])
# clf = clf.fit(train_text.astype(str),targets)

# num_cross = 10
# scores = cross_validation.cross_val_score(clf, train_text, targets, cv=5)

# print "Cross validation on", num_cross,"splits"
# print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)