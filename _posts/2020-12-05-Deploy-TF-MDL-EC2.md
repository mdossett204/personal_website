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
As mentioned earlier, this Flask application is deployed inside an Amazon EC2 linux2 instance. This blog assumes you already know how to deploy an EC2 instance. If you are new to EC2, please check out [Tutorial: Getting started with Amazon EC2 Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html). 

After successfully launch an EC2 instance, make sure you allocate an Elastic IP address and associate that IP address with your EC2 instance to ensure static IP address in case of emergency shutdown or resource outages. 

Once the IP address for your EC2 instance is constant, ssh into the instance to set up your Flask Application:

First, move your Flask application into the instance via scp or fetch from your remote git repository (if the application is in a git repository). Run the following command in bash to update your system, install python3 along with the required python packages. In the command shown below, the requirements.txt was made previously using Pipenv and saved in the Flask application folder. For detailed instructions on how to create requirements.txt using Pipenv click [here](https://pipenv-fork.readthedocs.io/en/latest/basics.html).

    sudo yum update -y  
	sudo yum install python3.7
	sudo -H pip3 install --upgrade pip
	sudo -H python3 -m pip install -r requirements.txt --no-cache-dir
	
Before moving onto NGINX, make sure your application runs as expected in the instance. Go to the application folder and run the Flask application. To make sure the application runs as expected, open a browser and type in the elastic IP address for the instance and append port number for the Flask application. 


![NGINX](/assets/img/blog6_img2.jpg){:class="post-image img-blur"}

#### NGINX And Systemctl
Once you confirm the Flask application runs as expected inside the instance, it is time to use NGINX to set the application as a servie. Use the following bash command to install and edit the NGINX configuration file. 

    sudo amazon-linux-extras install nginx1 
	cd /etc/nginx
	sudo vi nginx.config

Once you are inside the configuration file, use `/` to search for `server block for listen 80` inside the file. Make sure to update the `server_name` from `_` to the domain name of the Flask application (the CNAME for your dns)  and add the following JSON code to the `server block for listen 80`:

    location/{  
	       proxy_pass http://localhost:<application port#>; 
	}

Systemctl is used to run the Flask application as NGINX service in the background. 

First, use the bash command shown below to make sure the NGIX.service is running properly:

    sudo systemctl sart nginx.servie  
    sudo systemctl status nginx.service 
    
Once we make sure the nginx.service is actually running. Go back to the Flask application folder and run the application again. At this point, if you go to the instance IP address with the application port number, the Flask application should be running. 

To set up the application as a service running in the background, create an <application name>.service file inside /etc/systemd/system folder. 

Add the following lines to the <application name>.service file:

    [unit]
	Description=<Application Name>
	After=network.target
	StartLimitIntervalSec=0  
	
	[Service]                                                                                                                     
	Type=simple
	Restart=always
	RestartSec=1
	User=ec2-user
	WorkingDirectory=<Flask application folder path>
	ExecStart=/bin/python3 <Flask application python file name>  
	
	[Install]
	WantedBy=multi-user.target
	
Type in the following command in bash to reload the service files, enable both NGIX and flask applications services, and start the Flask application as a service in the background 

    systemctl daemon-reload  
    systemctl enable <Application Name>.service
    systemctl enable nginx
    systemctl start <Application Name>.service

To double check everything is running as expected, type the line below:

    systemctl status <Application Name>.service
    
When you go to the application port number of the EC2 elastic ipV4 address in a browser, the application should be running. 
    
#### Allow SSL on EC2 
Because the front end of this Flask application is hosted via https using S3 and AWS CloudFront, to allow SSL on EC2 as the back end of the application is an requirement. Fortunately Certbot makes the process relatively painless. For detailed information, checkout [Certifcate automation:Let's Encrypt with Certbot on Amazon Linux 2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/SSL-on-amazon-linux-2.html#letsencrypt). 

In a nut shell, type the following bash commands to install certbot:

    sudo wget -r --no-parent -A \
    'epel-release-*.rpm' \
    https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/
      
    sudo rpm -Uvh \
    dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-*.rpm
      
    sudo yum-config-manager --enable epel*

    sudo yum install -y certbot python2-certbot-nginx
    
To get the certificate, run the command below:

   sudo certbot --ngix -d <application dns name> --debug
   
If you want to set up an auto renewal of the certificate, you can use the certbot-renew.timer that is already installed in the instance. You can double check by typing the following:

    ls /usr/lib/systemd/system/ | grep certbot
    
If you see certbot-renew.service and certbot-renew.timer inside the folder, then it means the timer is already installed. Simply edit the following file and change the post_hook variable to the line shown below:

    vi /etc/sysconfig/certbot  
    POST_HOOK="--post-hook 'systemctl restart nginx'"
    
The default timer checks for certificate renewal eligibility twice a day.

Finally enable the auto renewal using systemctl and check if it is enabled by typing the commands below:

    sudo systemctl enable certbot-renew.timer
    systemctl is-enabled certbot-renew.timer 
    
If the enabled is shown, then the automatic renewal is set up properly. 

![certbot](/assets/img/blog6_img3.png){:class="post-image img-blur"}


## Conclusions
In this blog, we went over the basic steps to deploy a machine learning model prediction application as a background service in Amazon EC2 instance. Feel free to contact me with any questions you may have.  
