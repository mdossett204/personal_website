---
layout: post
title:  StopWords and Lexicon Normalization for Sentiment Analysis
date:   2020-10-21 10:58:00
image:  '/assets/img/02.jpg'
tags:   [NLP, stopwords, lexicon normalization, sentiment analysis, data science, AI]


Sentiment analysis aids in understanding human behavior for national security, political campaigns, and scientific research. It also allows businesses to gain valuable marketing information from customer reviews, social media sites, online blogs, and various internet forums. Not surprisingly, sentiment analysis is a very common yet crucial natural language processing (NLP) task that many data scientists are required to perform.

We will use [TextBlob](https://github.com/sloria/TextBlob/blob/eb08c120d364e908646731d60b4e4c6c1712ff63/textblob/_text.py) and [VADER](https://github.com/cjhutto/vaderSentiment) perform sentiment analysis on Yelp restaurant reviews in the D.C. Metro Area. The goals of this blog are to compare the sentiment polarity scores using both libraries with the actual Yelp food review ratings and to understand why removing StopWords and Lemmatization are not always necessary for [text analytics](https://opendatagroup.github.io/data%20science/2019/03/21/preprocessing-text.html).

![]({{site.baseurl}}/assets/img/05.jpg)


