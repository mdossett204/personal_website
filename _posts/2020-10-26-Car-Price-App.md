---
layout: post
title:  Deploying Car Price Model 
date:   2020-10-26 14:50:35 +0300
image:  '/assets/img/04.jpg'
tags:   [Regression, ML, AI, App, Car Price, prediction]
---

This post deploys a car price prediction Flask app using an ensemble of a deep learning regressor with categorical embeddings and a random forest regressor. This application is hosted using AWS EC2 instance. Due to the large size of tensorflow 2.2.0 package, a serverless approach of using lambda and API gateway is not practical.

For detailed machine learning/deep learning information, checkout this [GitHub link](https://github.com/mzhou356/CarPriceRegression).

For detailed information regarding the Flask app, checkout this [GitHub Link](https://github.com/mzhou356/Car-Price-Model-Application/tree/main/Flask_app).


![AWS EC2](/assets/img/blog3_img1.png){:class="post-image img-blur"}.  


## Key Concepts

This Flask application takes a single user input with pre-defined car related parameters. It predicts the estimated price by averaging the results from the deep learning model and the random forest model. This application runs as a systemd service using NGINX as a proxy inside an Amazon Linux2 instance.

Because this personal website is hosted via https using S3 and AWS CloudFront, this requires the backend application to also enable https. Certbot and Let's Encrypt are used to enable https on the EC2 instance. More detailed implementations will be posted in the future blogs. 

![AWS S3 and CloudFront](/assets/img/blog3_img2.png){:class="post-image img-blur"}. 