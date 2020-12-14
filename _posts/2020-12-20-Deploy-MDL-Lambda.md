---
layout: post
title:  Deploy Machine Learning Models using AWS Lambda and API Gateway
date:   2020-12-20 13:30:35 +0300
image:  '/assets/img/07.jpg'
tags:   [AWS, Lambda, API Gateway, EFS, serverless deploy model, cloud compute]
---

Intro part on the blog on how to deploy the random forest only model approach using AWS lambda with lambda layer and API gateway. 

![serverless using API gateway and lambda](/assets/img/blog7_img1.png){:class="post-image img-blur"}



## Lambda and Lambda Layers

One thing to note is that the large part of the python programming is actually in the `utils.py` file in [this flask_app folder](
https://github.com/mzhou356/Car-Price-Model-Application/tree/main/flask_app).

![lambda and lambda layers](/assets/img/blog7_img2.png){:class="post-image img-blur"}


## EFS and Lambda 
As mentioned earlier, this Flask application is deployed inside an Amazon EC2 Linux 2 

![Lambda and EFS](/assets/img/blog7_img3.png){:class="post-image img-blur"}

### EFS and EC2
Need to use EC2 to mount the packages onto EFS.
![EC2 and EFS](/assets/img/blog7_img4.jpg){:class="post-image img-blur"}

    
## API Gateway and Lambda 
Use api gateway to trigger lambda and enable CORS. 

### CORS

![CORS](/assets/img/blog7_img5.png){:class="post-image img-blur"}


## Conclusions
In this blog, we went over the basic steps to deploy a machine learning model prediction application in a serverless manner using AWS lambda and API gateway.
