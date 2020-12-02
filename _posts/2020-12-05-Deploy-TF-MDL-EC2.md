---
layout: post
title:  Deploy Machine Learning Models in AWS EC2
date:   2020-11-25 13:30:35 +0300
image:  '/assets/img/06.jpg'
tags:   [AWS, EC2, deploy model, cloud compute]
---

Although I am a big believer of serverless architecture, EC2 instances sometimes may be an easier option for deploying complicated machine learning models. 

My [car price model application](https://mindy-dossett.com/2020/10/26/Car-Price-App/) predicts a used car price by averaging the predicted prices from a random forest model and a deep learning model with categorical embeddings. Due to the large size of the TensorFlow package (> 1GB), it is much more feasible to deploy this model ensemble via an EC2 instance than lambda. I will have a separate blog on how to deploy the random forest only model approach using AWS lambda with lambda layer and API gateway. 

![tensorflow](/assets/img/blog6_img1.png){:class="post-image img-blur"}

This blog will be focused on how to deploy an already trained model using Python3 Flask and AWS EC2. 

## Flask App  
The Flask framework is used in this blog. The detailed python code can be found in [this github link](https://github.com/mzhou356/Car-Price-Model-Application/blob/main/flask_app/app.py).


One thing to note is that the large part of the python programming is actually in the utils file in [this flask_app folder](
https://github.com/mzhou356/Car-Price-Model-Application/tree/main/flask_app).

For this specific application, we only used request, jsonify from Flask to host the predict function. The request is used to get the user input from the front end as a json object. The utils file converts the front end user input into numpy arrays for model predictions. The final prediction score is returned as a json object via Flask jsonify. POST method is used for this application since user is sending information (used car inputs) to the backend to get a response (predicted price) back. 

Finally, Flask after_request is used to add headers to enable Cross-origin-resource-sharing(CORS). Once everything ran smoothly, we switched to waitress to run the application in production.


## EC2 





![NGINX](/assets/img/blog6_img2.jpg){:class="post-image img-blur"}

#### NGINX


#### Run as a service 


#### Allow SSL on EC2 

![certbot](/assets/img/blog6_img3.png){:class="post-image img-blur"}


## Conclusions
