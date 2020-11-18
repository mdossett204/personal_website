---
layout: post
title:  Common Methods for Tackling Imbalanced Datasets
date:   2020-11-17 13:30:35 +0300
image:  '/assets/img/05.jpg'
tags:   [data science, machine learning, imbalanced datasets, machine learning, classification]
---

One of the common machine learning classification challenges we face is an imbalanced data set. It can happen for binary and multi-class classification tasks. Often times, you will have one majority class (e.g. 80% of the label distribution) and one or more minority classes (e.g. 20% or 15% and 5 % of the label distribution). Fraud detection, special medical therapy, and equipment failure are examples of real-world class imbalance applications. This blog will go over some common tactics to help mitigate imbalanced data sets.

## Choose metrics carefully

Accuracy tends to be one of the most common metrics for classification problems. In the case of imbalanced data sets, you can easily achieve a very high accuracy by predicting every case to be the majority class. A confusion matrix, especially for multi-class classification problems is usually one of the best metrics to assess model performance. Depending on the application, for binary classification, recall (sensitivity), precision, or F1-score could be a better metric indicator. In general, a high recall and a high precision is what we want for any model. A low recall and a high precision means the model can't detect the class very well but each positive prediction is highly trustworthy. A high recall and a low precision suggests the class can be detected well by the model but many of the positive predictions are actually false.

![confusion matrix](/assets/img/blog4_img1.png){:class="post-image img-blur"}

## Data Resampling

If your data set contains many observations, it may be better to downsample the majority class. Smaller data sets may work better with upsampling. Generally speaking, it is a good idea to test out various class ratios instead of just a 1-to-1 ratio for all classes during up or downsampling. You can also use Synthetic Minority Over-sampling Technique (SMOTE) to synthetically increase minority class distribution. Generally speaking, it is very fast to test out all three approaches, so it can be a great starting point to try all 3 methods. It is important to note that resampling can be problematic. On one hand, minority class oversampling can lead to overfitting by pooling repeated data points from a very small data set. On the other hand, truncating data from the majority class can remove important information among the majority and minority classes. Even though SMOTE can improve oversampling by creating synthetic data points without duplicating data, it can still cause overfitting. Another problem with rebalancing the classes is that it may not work very well for future data sets because we show the wrong ratios of the classes to the models during training.

![SMOTE](/assets/img/blog4_img2.png){:class="post-image img-blur"}

## Gather More data

In addition to resampling your data, it can be helpful to retrieve more data. It can help balance out your data set or it can provide more minority class data points to help improve data upsampling or SMOTE and reduce overfitting issues. In addition, if possible, try to get additional attributes that can better differentiate the classes.

## Play with various machine learning algorithms. 

Generally speaking, it is a good idea to experiment with many different types of algorithms on any data set. A few great algorithms for imbalanced data sets are decision tree and random forest. In addition to trying various algorithms, testing out various class weights can help improve model performance. Another approach is cost sensitive learning where you penalize classification by adding a cost for making prediction mistakes on minority classes more heavily than majority classes during cross validation. You can also change probability thresholds for classification predictions based upon your desired metrics.

## Restate Your Question

Sometimes your business goals can't be solved from a basic classification perspective but rather can be treated as an anomaly detection or change detection problem. Both approaches treat the majority class as the norm and the minority class as the outliers or a change/difference from the norm. The new approach can offer alternative techniques to better tackle the problem. A few common methods are one-class support vector machines, cluster analysis outlier detection, and isolation forests.

![One Class SVM](/assets/img/blog4_img3.png){:class="post-image img-blur"}


## Conclusions
Class imbalance challenges are common among machine learning classification problems. It is important to gather as much data as possible and to find features that can better separate your classes. Picking the correct model metrics for your application is crucial for successful model tuning. You should always use resampling methods with caution. Don't only stick with the machine learning algorithms you know well but try to test out different models with various class weights, prediction penalties, and probability thresholds. Finally, sometimes it might be better to treat the classification as anomaly detection or change detection problems.