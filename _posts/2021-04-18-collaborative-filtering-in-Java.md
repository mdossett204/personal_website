---
layout: post
title:  A Collaborative Filtering Movie Recommender in Java
date:   2021-04-18 10:30:35 +0300
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

There are a few scores to measure user to user or item to item similarity. Pearson similarity and cosine similarity are the two most commonly used metrics for collaborative filtering algorithm. 

Cosine similarity score is the chosen metric in this blog. It is the simplest algorithm to find the similarity of two vectors as shown in the image below. 

![cosine similarity](/assets/img/blog9_img4.png){:class="post-image img-blur"}

The user score vector (row) or the item score vector (column) can be extracted from the user-item interaction matrix shown earlier. Generally speaking, we will encounter a very sparse matrix and a score of 0 will be assigned when there isn't a rating in the matrix. The cosine similarity score of -1 implies that the users or the items are polar opposites while a score of 1 shows that the users or the items are twins. A score of 0 indicates that the users or the items are completely unrelated. 

## User Based Collaborative Filtering 

This filtering algorithm recommends future items based upon assessing user to user similarity score. The formula for cosine similarity score between user a and b are shown below:

$$\mathcal{Sim(a,b)} = \frac{\displaystyle\sum_{p}(r_{ap}-\overline{r_a})(r_{bp}-\overline{r_b})}{\displaystyle\sqrt{\sum_{p}(r_{ap}-\overline{r_a})^2}\sqrt{\sum_{p}(r_{bp}-\overline{r_b})^2}}$$

`p` in the formula represents all of the commonly rated products by both user a and b. The average rating for both users are subtracted from each rating score to remove any underlying user bias. 

![user to user similarity](/assets/img/blog9_img2.png){:class="post-image img-blur"}

The detailed Java implementation for calculating the similarity score can be found [here](https://github.com/mzhou356/SimpleJavaRecommender/blob/main/Recommender/src/main/java/com/mindy/dossett/UserBasedRating.java). The `getUserSimilarity` and `calUserSim` methods calculate the cosine similarity scores for the target user and all other users in the database if there are at least 2 commonly rated products. The public method, `getSimilarRatings`, collects all of the similarity scores, takes into consideration only the top `numNeighbors` most similar users, and returns the predicted scores in descending order for movies with `minimalRater` or more users. The predicted rating score is calculated using this formula:

$$\mathcal{r_{ap}} = \overline{r_a}+\frac{\displaystyle\sum_{j\in users}\mathcal{Sim(a,j)}\ast (r_{jp}-\overline{r_j})}{\displaystyle\sum_{j\in users}| \mathcal{Sim(a,j)} |}$$


## Item Based Collaborative Filtering

The item based filtering suggests future products by calculating item to item similarity score. The cosine similarity formula is similar to the user based score: 

$$\mathcal{Sim(a,b)} = \frac{\displaystyle\sum_{u}(r_{ua}-\overline{r_u})(r_{ub}-\overline{r_u})}{\displaystyle\sqrt{\sum_{u}(r_{ua}-\overline{r_u})^2}\sqrt{\sum_{u}(r_{ub}-\overline{r_u})^2}}$$

`u` in the above formula stands for all of the common users for item a and b. Again to remove the underlying user bias, the average rating for each common user is subtracted from the item score. The equation for the item based prediction score is also very similar except that the average item score is used to estimate the predicted rating. This can cause the rating to go slightly above the max score of 10 but this approach allows us to remove any underlying movie bias. 

$$\mathcal{r_{ua}} = \overline{r_a}+\frac{\displaystyle\sum_{j\in items}\mathcal{Sim(a,j)}\ast (r_{uj} - \overline{r_j})}{\displaystyle\sum_{j\in items}| \mathcal{Sim(a,j)} |}$$

The detailed Java implementation for both formulae can be found [here](https://github.com/mzhou356/SimpleJavaRecommender/blob/main/Recommender/src/main/java/com/mindy/dossett/ItemBasedRating.java). The `getSimilarityScores` and `calItemSim` methods calculate the cosine similarity scores for each previously rated movie by the target user and all other movie items in the database if there are at least `minimalRater` commonly rated users for each movie pair. The public method, `getSimilarRatings`, collects all of the similarity scores as a HashMap, only calculates the predicated ratings for items with `neighborSize` or more movies, and returns the scores in descending order.
 
![item to item similarity](/assets/img/blog9_img3.png){:class="post-image img-blur"}

## Java Implementation of Both Filters

#### New Users

One challenge for most recommendation system is new users without any rating information. The [`UserInfoInitializer`](https://github.com/mzhou356/SimpleJavaRecommender/blob/main/Recommender/src/main/java/com/mindy/dossett/UserInfoInitializer.java) class solves this problem by creating a new user in the [`UserDatabase`](https://github.com/mzhou356/SimpleJavaRecommender/blob/main/Recommender/src/main/java/com/mindy/dossett/UserDatabase.java) and asks the new user to rate 20 randomly sampled movies. These movies are selected from the top 50 most commonly rated and the top 50 highest rated for any movies that are made in 1980 or later from the [`MovieDatabase`](https://github.com/mzhou356/SimpleJavaRecommender/blob/main/Recommender/src/main/java/com/mindy/dossett/MovieDatabase.java). A score of -1 is given to unseen movies, so the rating information is not added to the `UserDatabase`. 

![Java](/assets/img/blog9_img5.jpeg){:class="post-image img-blur"}

#### Recommendation

The [`RecommenderRunner`](https://github.com/mzhou356/SimpleJavaRecommender/blob/main/Recommender/src/main/java/com/mindy/dossett/RecommenderRunner.java) class calls the `UserInfoInitializer` and adds the rating information to the `UserDatabase` class. It also calls both the `UserBasedRating` and `ItemBasedRating` to retrieve two lists of recommended movies from both filters. Instead of combining the lists and recommending all of the products, only the 10 highest rated movies from both filters are combined and reordered in descending fashion based upon the predicted score. Finally, the 10 overall highest rated movies are shown to the new user. 

## Summary
In this blog, we went over the basic algorithm of collaborative filtering for both user and item based similarity. We also shared detailed Java implementation of both for a simple movie recommendation program. Feel free to clone the [Java program](https://github.com/mzhou356/SimpleJavaRecommender) and try out the application on your local computer. 