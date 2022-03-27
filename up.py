#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:15:16 2022

@author: stone
"""

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy import API
from tweepy import Cursor

from textblob import TextBlob
import cred

import re
import numpy as np
import pandas as pd

import smtplib

import matplotlib.pyplot as plt


class Query():

    def __init__(self, stock,search_num, fig):
        self.stock = stock
        self.tweets = []
        self.search_num = search_num
        self.fig = fig
        self.pos = 0.0
        self.neg = 0.0
        self.neu = 0.0
        self.tot = 0

    def get_tweets(self):
        auth = OAuthHandler(cred.API_KEY, cred.API_SERCRET_KEY)
        auth.set_access_token(cred.ACCESS_TOKEN, cred.ACCESS_TOKEN_SECRET)
        api = API(auth, wait_on_rate_limit=True)
        for tweet in Cursor(api.search, q=self.stock, tweet_mode='extended').items(self.search_num):
            self.tweets.append(tweet)

        #[print(tweet.full_text, tweet.favorite_count, tweet.retweet_count) for tweet in self.tweets]

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def percentage(self, part, whole):
        temp = 100 * float(part) / float(whole)
        return format(temp, '.2f')

    def sentiment(self,tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity < 0:
            return -1
        else:
            return 0
        
    def plotPieChart(self, positive, negative, neutral):
        labels = ['Positive [' + str(positive) + '%]',
                   'Neutral [' + str(neutral) + '%]',
                  'Negative [' + str(negative) + '%]',
        ]
        sizes = [positive, neutral, negative ]
        plt.figure(self.fig)
        colors = [ 'gold','lightsalmon', 'red']
        explode = (0.05, 0.05, 0.05)
        patches, texts = plt.pie(sizes, colors=colors, startangle=90, explode = explode)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + self.stock + ' by analyzing ' + str(self.search_num) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        
    def get_data(self):
        df = pd.DataFrame(data=[self.clean_tweet(tweet.full_text) for tweet in self.tweets ], columns = ["Tweets"])
        df['Likes'] = np.array([tweet.favorite_count for tweet in self.tweets])
        df['Retweets'] = np.array([tweet.retweet_count for tweet in self.tweets])
        df['Sentiment'] = np.array([ self.sentiment(tweet) for tweet in df['Tweets']])
        df['User'] = np.array([tweet.user.name for tweet in self.tweets])
        df['Location'] = np.array(user.location for user in np.array([tweet.user for tweet in self.tweets]))
        #np.array([tweet.user.name for tweet in self.tweets])
        pos = 0
        neg = 0
        neu = 0
        tot = len(df['Sentiment'])
        sentt = []
        i = 0
        for sent in df['Sentiment']:
            sentt.append(sent)
            if sent == 1:
                pos += 1
            elif sent == -1:
                neg += 1
            else:
                neu += 1
        
        
        for loc in df['Location']:
            print(loc)
            
        print(df.head)
        df.to_csv (r'/Users/stone/Dropbox/stock/tweets.csv', index = False, header=True)
    
        #refugee, children, food storage, bombing, hopital, banks, food pantry, winter clothes, child education
        # crop production, food production, corn production, sunflower seed, sunflower oil, used in fastfood indusrty
        #for tweet in df["Tweets"]:
            #print(tweet, sentt[i])
            #i = i + 1

       # print(df.head(self.search_num))
        self.pos = self.percentage(pos,tot)
        self.neg = self.percentage(neg,tot)
        self.neu = self.percentage(neu,tot)
        self.plotPieChart(self.pos,self.neg,self.neu)
        

keywords = ['#ukraine']
tweet_objects = []
figure = 0
for word in keywords:
    tweet_data = Query(word,200,figure)
    tweet_data.get_tweets()
    tweet_data.get_data()
    figure += 1
    tweet_objects.append(tweet_data)