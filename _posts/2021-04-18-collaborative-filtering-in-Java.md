---
layout: post
title:  A Collaborative Filtering Movie Recommender in Java
date:   2021-04-15 13:30:35 +0300
image:  '/assets/img/09.jpeg'
tags:   [recommendation system, cosine similarity, collaborative filtering, Java, item to item similarity, user to user similarity]
mathjax: true
---

## Introduction

Collaborative filtering is one of the most popular recommendation engine algorithms for suggesting new products to customers. Today we are going to implement both user and item based collaborative filtering in Java and apply the similarity scores from both filters to recommend movie products to new users. 

## Collaborative Filtering

![movie and user rating matrix](/assets/img/blog9_img1.png){:class="post-image img-blur"}

Collaborative Filtering uses an user-item interaction matrix shown above to extract similarity scores among items or among users. The underlying assumption is:

> User based collaborative filtering: users that tend to enjoy many common items would most likely agree on new products. 

> Item based collaborative filtering: new items that are similar to other highly rated products by the target user would be a good recommendation.  


## Cosine Similarity Score

There are a few scores to measure user to user or item to item similarity. Pearson similarity and cosine similarity are the two most commonly for collaborative filtering algorithm. 

Cosine similarity score is the chosen metric in this blog. It is the simplest algorithm to find the similarity of two vectors as shown in the image below. 

![cosine similarity](/assets/img/blog9_img4.png){:class="post-image img-blur"}

The user score vector (row) or the item score vector (column) can be extracted from the user-item interaction matrix shown earlier. Generally speaking, we will encounter a very sparse matrix and a score of zero will be assigned when there isn't a rating in the matrix. The cosine similarity score of -1 implies that the users or the items are complete polar opposites while a score of 1 shows that the users or the items are twins. A score of 0 indicates that the users or the items are completely unrelated. 

## User Based Collaborative Filtering 

This filtering algorithm recommends future items based upon assessing user to user similarity score. The formula for cosine similarity score between user a and b are shown below:

$$\mathcal{Sim(a,b)} = \frac{\displaystyle\sum_{p}(r_{ap}-\overline{r_a})(r_{ab}-\overline{r_b})}{\displaystyle\sqrt{\sum_{p}(r_{ap}-\overline{r_a})^2}\sqrt{\sum_{p}(r_{bp}-\overline{r_b})^2}}$$



![user to user similarity](/assets/img/blog9_img2.png){:class="post-image img-blur"}

Once similarity score is estimated, a predicted score is calculated using this formula:

$$\mathcal{r_{ap}} = \overline{r_a}+\frac{\displaystyle\sum_{j\in users}\mathcal{Sim(a,j)}\ast (r_{jp}-\overline{r_j})}{\displaystyle\sum_{j\in users}| \mathcal{Sim(u,j)} |}$$

## Item Based Collaborative Filtering

Item based filtering suggests future items by calculating item to item similarity score. The cosine similarity formula is similar to the user based score: 

$$\mathcal{Sim(a,b)} = \frac{\displaystyle\sum_{u}(r_{ua}-\overline{r_u})(r_{ub}-\overline{r_u})}{\displaystyle\sqrt{\sum_{u}(r_{ua}-\overline{r_u})^2}\sqrt{\sum_{u}(r_{ub}-\overline{r_u})^2}}$$



![item to item similarity](/assets/img/blog9_img3.png){:class="post-image img-blur"}

The equation for the item based prediction score is slightly different:

$$\mathcal{r_{ua}} = \overline{r_a}+\frac{\displaystyle\sum_{j\in items}\mathcal{Sim(a,j)}\ast (r_{uj} - \overline{r_j})}{\displaystyle\sum_{j\in items}| \mathcal{Sim(a,j)} |}$$



In the example shown above, each input feature only has one weight parameter and they are only being fed into one output neuron unit. In the case of a fully connected neural network as shown in the image below, each input feature can be fed into one or more hidden layers with multiple hidden units. This means each input has multiple weight parameters for extrapolating latent feature information. This is why a logistic regression model can be thought as a fully connected neural network with just an input layer and a one neuron output layer.  



As mentioned above, a fully connected neural network is basically a logistic regression but with multiple hidden layers and hidden neuron units. The output layer can have multiple neurons as well depending upon the application. Each initial feature input is being fed into all hidden units in the first hidden layer and can have multiple weights associated with different hidden information. The interpretation of these transformed features are not always obvious. For example, if you are using a fully connected neural network to predict if a product should be marketed to an online customer, you may have data such as gender, age, address, and some other basic information as input features. The hidden units will be able to use this set of basic inputs to extract latent features. These latent features will often be more predictive than those that can be extracted through manual feature engineering. Often times, we can fed the transformed first hidden layer features into more hidden layers to extrapolate more complex hidden feature information. This is why some of us equate deep learning to artificial intelligence. We don't know exactly how each input maps to the hidden feature spaces, but with enough data to calculate millions of parameters, the model will be able to figure out the intermediate feature information and outperform traditional machine learning models. 
 
As I have already mentioned, a fully connected neural network can have millions of parameters (or more!). We need enough labelled data to statistically approximate these parameters. This is why deep learning really shines when we have access to a large quantity of labelled data. In applications where labelled data is limited, innovative feature engineering with a basic logistic regression or other machine learning models can often outperform a neural network. 

    
## Java Implementation
Generally speaking, deep learning can be quite powerful for supervised learning especially when there are large amounts of labeled data available. It is great for both structured and unstructured data. Deep learning also **reduces** the need for creative feature engineering. 

## Summary