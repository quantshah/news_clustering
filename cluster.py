from __future__ import print_function
import re
import random
import csv
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
from time import time
from scipy.cluster.hierarchy import ward, dendrogram


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
training_text = data['title'] 
# tagged_text = training_text.map(lambda x : getHighInfoWords(x))
train_text = training_text.map(lambda x : removePunctuation(x))

target = data['category']
dataset = train_text

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()

vectorizer = TfidfVectorizer(max_df=0.8,
                                 min_df=2, ngram_range=(1,3),
                                 )
                                 
tfidf_matrix = vectorizer.fit_transform(dataset)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % tfidf_matrix.shape)

dist = 1 - cosine_similarity(tfidf_matrix)


linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right",labels=training_text.tolist());

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()