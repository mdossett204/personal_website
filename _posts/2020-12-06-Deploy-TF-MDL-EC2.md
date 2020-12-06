---
layout: post
title:  Deploy Machine Learning Models in AWS EC2
date:   2020-12-06 13:30:35 +0300
image:  '/assets/img/06.jpg'
tags:   [AWS, EC2, deploy model, cloud compute]
---

Although I am a big believer of serverless architecture, EC2 instances sometimes may be an easier option for deploying complicated machine learning models. 

My [car price model application](https://mindy-dossett.com/2020/10/26/Car-Price-App/) predicts a used car price by averaging the predicted prices from a random forest model and a deep learning model with categorical embeddings. Due to the large size of the TensorFlow package (> 1GB), it is much more feasible to deploy this model ensemble via an EC2 instance than lambda. I will have a separate blog on how to deploy the random forest only model approach using AWS lambda with lambda layer and API gateway. 

![tensorflow](/assets/img/blog6_img1.png){:class="post-image img-blur"}

This blog will be focused on how to deploy an already trained model using Python3, Flask, and AWS EC2. 

## Flask App  
The Flask framework is used in my application. The detailed python code can be found in [this github link](https://github.com/mzhou356/Car-Price-Model-Application/blob/main/flask_app/app.py).


One thing to note is that the large part of the python programming is actually in the `utils.py` file in [this flask_app folder](
https://github.com/mzhou356/Car-Price-Model-Application/tree/main/flask_app).

For this specific application, we only used a few functions from the Flask library to host the prediction function, mainly `request` and `jsonify`. As usual, `request` is used to get the user input which is sent as a json object from the front end using the HTTP POST method. The `utils.py` file converts the front end user input into numpy arrays for model predictions. The final prediction score is returned as a json object using `jsonify`.  This function automatically adds the necessary headers to the response object.  

Finally, Flask `after_request` decorator is used to add headers to enable Cross-origin-resource-sharing(CORS) in the application. Once everything ran smoothly, we used `Waitress` rather than Flask's built-in http server to run the application in production.


## EC2 
As mentioned earlier, this Flask application is deployed inside an Amazon EC2 Linux 2 instance. This blog assumes you already know how to deploy an EC2 instance. If you are new to EC2, please check out Amazon's [Tutorial: Getting started with Amazon EC2 Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html). 

After successfully launching an EC2 instance, make sure you allocate a public Elastic IP address and associate it with your EC2 instance to ensure a static IP address in case of an outage.  You can then ssh into the instance to set up your Flask Application:

First, move your Flask application folder into the instance via scp or fetching from your remote git repository. Run the following command in bash to: 1) update your system; and 2) install python3 along with the required python packages. In the command shown below, the `requirements.txt` was made previously using Pipenv and saved in the Flask application folder. For detailed instructions on how to create `requirements.txt` using Pipenv click [here](https://pipenv-fork.readthedocs.io/en/latest/basics.html).

    sudo yum update -y  
	sudo yum install python3.7
	sudo -H pip3 install --upgrade pip
	sudo -H python3 -m pip install -r requirements.txt --no-cache-dir
	
Before moving onto NGINX, make sure your application runs as expected in the instance.

![NGINX](/assets/img/blog6_img2.jpg){:class="post-image img-blur"}

#### NGINX And systemd
Once you confirm the Flask application runs as expected inside the instance, it is time to use NGINX as a web proxy and `systemd` to set the application as a servie. Use the following bash commands to install NGINX:

    sudo amazon-linux-extras install nginx1
	
Next, edit the NGINX configuration file: `/etc/nginx/nginx.config`. Go to the server block for standard http traffic (search for `listen 80`). Make sure to update the `server_name` from `_` to the domain name where you want to run the Flask application. Then add the following lines to the block:

    location / {  
	       proxy_pass http://localhost:<application port#>; 
	}

This routes traffic coming in to your EC2 instance on port 80 to your Flask application regardless of what port you are running it on.  

Systemd is used to run the Flask application as a service in the background. This will allow the application to automatically start with the system without any user interaction. 

First, use the bash commands shown below to make sure the pre-existing NGINX service is running properly:

    sudo systemctl start nginx.servie  
    sudo systemctl status nginx.service 
    
Once we make sure the NGINX service is actually running. Go back to the Flask application folder and run the application again. At this point, you can test requests to your Flask app using a standard HTTP request. Once you have confirmed NGINX is properly passing requests to your application, you can move on to the next step to set up your application as a background service. 

Now, create an `<application-name>.service` file inside /etc/systemd/system folder. 

Add the following lines to the `<application-name>.service` file:

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
	
Then, type in the following commands in bash to: 1) reload the service files; 2) enable both the NGINX and Flask application services; and 3) start the Flask application:

    systemctl daemon-reload  
    systemctl enable <application-name>
    systemctl enable nginx
    systemctl start <application-name>

You can use the line below to check the status of your application:

    systemctl status <application-name>.service
    
You can once again test your application using a standard HTTP request to your chose domain name.
    
#### Setting up HTTPS
Because the front end of this Flask application is hosted via HTTPS using S3 and AWS CloudFront, it is required that the prediction backend also have HTTPS enabled.  Fortunately, [Let's Encrypt](https://letsencrypt.org/) and [certbot](https://certbot.eff.org/) make the process relatively painless. For detailed information, checkout [Certifcate automation: Let's Encrypt with Certbot on Amazon Linux 2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/SSL-on-amazon-linux-2.html#letsencrypt). 

In a nutshell, type the following bash commands to install certbot:

    sudo wget -r --no-parent -A 'epel-release-*.rpm' \
    https://dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/
      
    sudo rpm -Uvh \
    dl.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-*.rpm
      
    sudo yum-config-manager --enable epel*

    sudo yum install -y certbot python2-certbot-nginx
    
To get the certificate for your domain (or subdomain), run the command below

   sudo certbot --nginx -d <your-(sub)domain-name> --debug
   
If you want to set up an auto renewal of the certificate (strongly encouraged), you can use the `certbot-renew` systemd timer that should already be installed in the instance. You can double check by typing the following:

    ls /usr/lib/systemd/system/ | grep certbot
    
If you see `certbot-renew.service` and `certbot-renew.timer` inside the folder, then it means the timer is already installed. Simply edit `/etc/sysconfig/certbot` and change the `POST_HOOK` variable to the line shown below:

    POST_HOOK="--post-hook 'systemctl restart nginx'"
    
This makes sure that NGINX loads the new certificate when it is issued.  

Finally, enable the auto-renewal using systemctl and check if it is enabled by typing the commands below:

    sudo systemctl enable certbot-renew.timer
    systemctl is-enabled certbot-renew.timer   

If the `enabled` is shown, then the automatic renewal is set up properly.  By default, the timer will check for certificate renewal eligibility twice a day.

![certbot](/assets/img/blog6_img3.png){:class="post-image img-blur"}


## Conclusions
In this blog, we went over the basic steps to deploy a machine learning model prediction application as a background service in an Amazon EC2 instance using Flask, NGINX, systemd and certbot. Feel free to contact me with any questions you may have.  
