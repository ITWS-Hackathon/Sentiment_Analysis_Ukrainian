import numpy as np
import pandas as pd
import nltk

import os

df_review = [line.rstrip() for line in open('tweets.csv')]
print (len(df_review))
import pandas
df_review = pandas.read_csv('tweets.csv', sep='\t')
df_review.head()
df_review.describe()
df_review.groupby('rating').describe()
df_review['length'] = df_review['verified_reviews'].apply(len)
df_review.head()
import matplotlib.pyplot as plt
import seaborn as sns
df_review['length'].plot(bins=50, kind='hist')
df_review.length.describe()
df_review[df_review['length'] == 2851]['verified_reviews'].iloc[0]
df_review.hist(column='length', by='feedback', bins=50,figsize=(10,4))
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('../tweets.csv', delimiter = '\t', quoting = 3)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,3150):
    review = re.sub('[^a-zA-Z]', ' ', dataset['verified_reviews'][i] )
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,4].values

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
nX = None
ny = None
for train_index, test_index in sss.split(X, y_feature):
        nX = X[test_index].copy()
        ny = y_feature[test_index].copy()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm