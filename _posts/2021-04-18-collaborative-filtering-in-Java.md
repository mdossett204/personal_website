---
layout: post
title:  A Collaborative Filtering Movie Recommender in Java
date:   2021-04-15 13:30:35 +0300
image:  '/assets/img/09.jpeg'
tags:   [recommendation system, cosine similarity, collaborative filtering, Java, item to item similarity, user to user similarity]
mathjax: true
---

## Introduction

There are three main types of neural networks:  
> 1. Fully Connected Neural Network (Dense Layer)
> 2. Convolutional Neural Network (CNN)
> 3. Recurrent Neural Network (RNN, LSTM, GRU) 

In this blog post we will give a gentle introduction to a fully connected neural network and explain why logistic regression models are basically fully connected neural networks with one output neuron unit and zero hidden layers. 

![movie and user rating matrix](/assets/img/blog9_img1.png){:class="post-image img-blur"}


## User Based Collaborative Filtering 

![user to user similarity](/assets/img/blog9_img2.png){:class="post-image img-blur"}

## Item Based Collaborative Filtering
As shown in the graph below, standard logistic regression classifiers typically have a set of input features, \\(x_0\\) to \\(x_{m}\\) and corresponding weights/coefficients, \\(w_0\\) to \\(w_{m}\\) where \\(m\\) stands for the number of input features. These terms, combined with a bias term, \\(b\\) are used to calculate the probability of a binary outcome after a nonlinear activation function called sigmoid. 

$$\mathcal{P} = \mathrm{sigmoid}(x_0w_0+\dotsb+ x_{m}w_{m} + b) $$



![item to item similarity](/assets/img/blog9_img3.png){:class="post-image img-blur"}

In the example shown above, each input feature only has one weight parameter and they are only being fed into one output neuron unit. In the case of a fully connected neural network as shown in the image below, each input feature can be fed into one or more hidden layers with multiple hidden units. This means each input has multiple weight parameters for extrapolating latent feature information. This is why a logistic regression model can be thought as a fully connected neural network with just an input layer and a one neuron output layer.  

## Cosine Similarity 
![cosine similarity](/assets/img/blog9_img4.png){:class="post-image img-blur"}

As mentioned above, a fully connected neural network is basically a logistic regression but with multiple hidden layers and hidden neuron units. The output layer can have multiple neurons as well depending upon the application. Each initial feature input is being fed into all hidden units in the first hidden layer and can have multiple weights associated with different hidden information. The interpretation of these transformed features are not always obvious. For example, if you are using a fully connected neural network to predict if a product should be marketed to an online customer, you may have data such as gender, age, address, and some other basic information as input features. The hidden units will be able to use this set of basic inputs to extract latent features. These latent features will often be more predictive than those that can be extracted through manual feature engineering. Often times, we can fed the transformed first hidden layer features into more hidden layers to extrapolate more complex hidden feature information. This is why some of us equate deep learning to artificial intelligence. We don't know exactly how each input maps to the hidden feature spaces, but with enough data to calculate millions of parameters, the model will be able to figure out the intermediate feature information and outperform traditional machine learning models. 
 
As I have already mentioned, a fully connected neural network can have millions of parameters (or more!). We need enough labelled data to statistically approximate these parameters. This is why deep learning really shines when we have access to a large quantity of labelled data. In applications where labelled data is limited, innovative feature engineering with a basic logistic regression or other machine learning models can often outperform a neural network. 

    
## Java Implementation
Generally speaking, deep learning can be quite powerful for supervised learning especially when there are large amounts of labeled data available. It is great for both structured and unstructured data. Deep learning also **reduces** the need for creative feature engineering. 

## Summary