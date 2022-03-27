import numpy as np
import pandas
import matplotlib.pyplot as plt
import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import StratifiedShuffleSplit

def polarity(tweet):
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity < 0:
        return -1
    else:
        return 0

X_data = pandas.read_csv(open('tweets.csv'), delimiter=",")
df_tweet = X_data['Tweets']
N = len(df_tweet)
corpus=[]
y_feature = []
#nltk.download('stopwords')
# Data Cleaning
pos = 0
neg = 0
neu = 0
for i in range(N):
    tweet = re.sub('[^a-zA-Z]', ' ', df_tweet[i]) # Remove anything that is not alphabets
    tweet = tweet.lower()
    tweet = tweet.split()
    ps = PorterStemmer()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    corpus.append(tweet)
    p = polarity(tweet)
    if p == 1:
        pos += 1
        y_feature.append(1)
    elif p == -1:
        neg += 1
        y_feature.append(-1)
    else:
        neu += 1
        y_feature.append(0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

X = np.asarray(corpus)
X = X.reshape(N, -1)
y_feature = np.asarray(y_feature)
# Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
for train_index, test_index in sss.split(X, y_feature):
    X_train = X[train_index].copy()
    y_train = y_feature[train_index].copy()
    X_test = X[test_index].copy()
    y_test = y_feature[test_index].copy()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)