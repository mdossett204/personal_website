---
layout: post
title:  Deploy Machine Learning Models using AWS Lambda and API gateway
date:   2020-12-31 13:30:35 +0300
image:  '/assets/img/07.jpg'
tags:   [AWS, Lambda, API gateway, serverless deploy model, cloud compute]
---

Today we will talk about how to deploy a Python machine learning model using AWS lambda with lambda layers and API gateway. The raw code can be found in this [Github link](https://github.com/mzhou356/Car-Price-Model-Application/tree/main/car_price_pred).


As shown in the image below, an end user will send a HTTPS POST request to API gateway, this will trigger the lambda function to perform model prediction and send the result back to API gateway and back to the end user in the front end. 

![serverless using API gateway and lambda](/assets/img/blog7_img1.png){:class="post-image img-blur"}

We will go through the key steps necessary to set up the lambda function and API gateway but the machine learning step will be ignored in this blog. 


## Lambda and Lambda Layers

### Lambda Deployment Package  
A prebuilt random forest model has been saved using `joblib` inside the Lambda deployment package folder along with `utils.py` and other pickle files necessary for model prediction. The actual lambda function code can be found [here](https://github.com/mzhou356/Car-Price-Model-Application/blob/main/car_price_pred/lambda_function.py). 

Below are a few important things to point out from the `lambda_function.py` file:  

1. Most of the actual Python code is inside the `utils.py` file. After a quick glance of the code, you will notice that we need to add `joblib`, `pandas`, and `scikit-learn` along  with `Numpy` and `SciPy` to the Lambda function package. You can find a complete list of available python modules available in `Boto3` from this [GitHub link](https://gist.github.com/gene1wood/4a052f39490fae00e0c3#file-all_aws_lambda_modules_python3-7-txt).

2. Make sure you update the Lambda Handler field in the settings to match your lambda function name (in this case, our function name is called `lambda_handler`). 

3. Argument `event` contains the HTTPS POST information from API gateway and the `json` library is imported to load the user_input as a `json` object. 

4. It is important to include `"headers":{"Content-Type":"application/json"}` in the return `json` object to get the prediction result back.

5. The zipped deployment package needs to be < 50MB. If you would like to view the code in the Lambda console, the package size needs to be < 3MB.  

### Lambda Layers

![lambda and lambda layers](/assets/img/blog7_img2.png){:class="post-image img-blur"}

At this point, you might wonder how we can deploy this model using Lambda given the 50MB restriction and the need to include additional Python modules mentioned earlier. Luckily, AWS allows us to use Lambda layers. The actual limit for both the deployment package and the layers unzipped is 250MB. Additionally, AWS already has a Lambda layer called `AWSLambda-Python37-Scipy1x` that contains `Numpy` and `Scipy`.

In order to add `scikt-learn` and `pandas` along with `joblib`, we will need to first create a file called `requirements.txt`. There are a few ways to make this, I simply used `Pipenv` to create this file. You can find the detailed information in this [blog](https://drgabrielharris.medium.com/python-how-create-requirements-txt-using-pipenv-2c22bbb533af).

Once we have the `requirements.txt`, you can either create the Lambda layer inside Amazon Linux 2 AMI or use docker to create the layers. If you don't use the same environment as the one AWS Lambda uses, your uploaded Lambda layer may not be compatible.

Since I already have docker installed on my system, I used a Docker to simulate the Lambda environment and created the layer. You can find the detailed steps here, [How do I create a Lambda Layer using a simulated Lambda environment with Docker?](https://aws.amazon.com/premiumsupport/knowledge-center/lambda-layer-simulated-docker/).

Before you publish the Lambda layer, make sure to customize the `pandas` and especially `scikt-learn` packages, or you will run into size limit issues when you add the Lambda layer to the function. There are a few ways we can reduce the module size:  

1. Remove `Numpy` and `Scipy` along with other preexisting modules that are part of the dependencies for either package.  

2. Remove any directories that are for testing only.

Before we move on, I do want to point out that you can probably include `joblib`, `pandas`, and `scikit-learn` in the zipped deployment package and stay barely under 50MB, but it is generally not a good idea to upload such a large zipped file directly to the Lambda console.

Once you successfully uploaded the zipped deployment package and added Lambda layers to your function, make sure to increase the Memory size to 512 MB. Set up a simple test first to make sure the function runs properly. Once it runs, we will set up the API gateway trigger in the section below. 

    
## API gateway and Lambda

You will head to API gateway in AWS console and create a REST API. We will select new API option and add a custom name to the API and leave the rest to be default. 

Next, we will create a post method and integrate our Lambda function to this API. Make sure we select `Use Lambda Proxy Integration` and the correct Region where our Lambda function resides. In the `Lambda Function` text box, when you start to type in the function name, it should autofill. After you click save, the trigger for Lambda and corresponding permissions should be set up automatically in the Lambda function. Test to make sure it works by clicking test and enter a test input in `json` format. It should get a predicted price back in this format: `{"predicted_price": <price>}`.

### CORS
Even though the post method worked, an end user from the web front end currently won't be able to send an HTTPS post method to API gateway without CORS enabled. It stands for Cross-origin resource sharing. You can read more about CORS [here](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS). 

![CORS](/assets/img/blog7_img3.jpg){:class="post-image img-blur"}

To enable CORS in the REST API, simply click Actions and select Enable CORS. Leave everything default and click `Enable CORS and replace existing CORS headers`. This should work theoretically. But you may still run into a CORS issue, one way that worked for me is to add the `'Access-Control-Allow-Origin':'*'` in your headers as part of the returned `json` object from the Lambda function.

Once everything is working, we are ready to deploy the API. Simply going to Actions and select Deploy. Add a new stage by giving a name and descriptions for stage and deployment. Once the stage is created, you will get an Invoke URL. You can always add a custom domain name to your API gateway. In my case, I simply copied and pasted the default URL to the JavaScript where it specifies the url for `ajax` post. 

Once the URL is updated, we should be able to go to the web front end, enter some inputs and get a predicted price back from AWS Lambda and API gateway. 

## Conclusions  
In this blog, we went over the basic steps to deploy a machine learning model in a serverless manner using AWS lambda and API gateway. Feel free to email me with any questions or any tips or tricks. 
