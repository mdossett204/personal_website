---
layout: post
title:  Deploy Machine Learning Models using AWS Lambda and API Gateway
date:   2020-12-12 13:30:35 +0300
image:  '/assets/img/07.jpg'
tags:   [AWS, Lambda, API Gateway, EFS, serverless deploy model, cloud compute]
---

Today we will talk about how to deploy a Python machine learning model using AWS lambda with lambda layers and API gateway. The raw code can be found in this [Github link](https://github.com/mzhou356/Car-Price-Model-Application/tree/main/car_price_pred).


As shown in the image below, an end user will send a HTTPS POST request to API Gateway, this will trigger the lambda function to perform model prediction and send the result back to API Gateway and back to the end user in the front end. 

![serverless using API gateway and lambda](/assets/img/blog7_img1.png){:class="post-image img-blur"}

We will go through the key steps necessary to set up the lambda function and API gateway but the machine learning step will be ignored in this blog. 


## Lambda and Lambda Layers

### Lambda Deployment Package  
A prebuilt random forest model has been saved using Joblib inside the Lambda deployment package folder along with `utils` file and other saved pickle files necessary for model prediction. The actual lambda function code can be found [here](https://github.com/mzhou356/Car-Price-Model-Application/blob/main/car_price_pred/lambda_function.py). 

Below are a few important things to point out from the `lambda_function.py` file:  

1. Most of the actual Python code are inside the `utils.py` file and if you look inside the code, you will notice that we will need to add `joblib`, `pandas`, and `scikit-learn` along with `Numpy` and `SciPy` to the Lambda function.  

2. Make sure you update the Lambda Handler field in the setting to match the lambda function (in this case, the name is called `lambda_handler`. 

3. Argument `event` contains the HTTPS POST information from API gateway and `json` library is needed to load the user_input as a json object. 

4. It is important to include `"headers":{"Content-Type":"application/json"}` in the return json object.

5. The zipped deployment package needs to be <50MB and to be able to view the code in the Lambda console, the file need to be < 3MB.  

### Lambda Layers

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

![CORS](/assets/img/blog7_img5.jpg){:class="post-image img-blur"}


## Conclusions
In this blog, we went over the basic steps to deploy a machine learning model prediction application in a serverless manner using AWS lambda and API gateway.
