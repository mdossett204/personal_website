---
layout: post
title:  "Transfer Learning for Social Media Marketing"
date:   2020-10-19 11:17:55 +0300
image:  '/assets/img/03.jpg'
tags:   Jekyll
---
Generally speaking, an average Instagram advertisement costs around $0.50 to $1 per click. In more competitive fields such as the fashion industry, it can cost up to $3 per click. The cost per click can also vary among age groups. For example, the 18–24, 25–34, and 35–44 age groups can cost more than others. Ad costs can also vary based on the targeted sex. Facebook ads targeting females tend to have a higher cost. These relatively high costs per click combined with the fact that not all clicks lead to sales can make ad campaigns on social media quite expensive.

Another popular internet marketing strategy is the celebrity product endorsement. An endorsement by an Instagrammer with 3 to 7 million followers can cost up to $75,000. Influencers with 50,000 to 500,000 followers can charge up to $1000 per endorsement. Celebrity endorsements can often lead to more sales than random ad clicks but the marketing cost is still relatively high. For this reason, I would like to introduce an IMG2RECOMMENDER concept for more targeted social media advertisement. This approach uses transfer learning to convert social media images to more targeted product advertisements. The general concept is shown in the chart below.

![Recommender Concept](/assets/img/blog1_img1.png){:class="post-image img-blur"}

We are in the era of sharing millions of images on social media. The most popular social media sites are Facebook, Instagram, and Pinterest. Instead of paying for non-targeted ads on these sites, companies can integrate a sophisticated convolutional neural network model (CNN) with an accurate product identifier as a back-end app to identify the most relevant products to each social media user. The marketing companies would only need to pay a monthly fee to these social media sites to run these back-end apps.

In addition to the obvious benefit of cost reduction, a few extra added benefits are:  
* Identify potential customers without additional social media accounts.
* Identify relevant products for personal usage and gifts to friends.
* Reduction in labor capital.
* No need for cross site tracking (increases personal privacy).
* No need for storing personal data for individual users.

I designed a 3 product recommender model as a proof of concept for this marketing app. Detailed code can be found in this [GitHub repository](https://github.com/mzhou356/img2recommender).

![Face Focused Concept](/assets/img/blog1_img2.png){:class="post-image img-blur"}. 

As shown in the figure above, I start with the open source face_recognition library to perform face cropping. Transfer learning models are then utilized to perform product predictions for more targeted advertisement. In this model, the recommended items are hat, beard, or eyewear products (HBE).

For this prototype, I manually web-scraped 653 images from google images. For training of my models, both image labeling and cropping are done manually as well. Generally speaking, it is very labor intensive to obtain adequate labeled image data for CNN projects. I decided to use transfer learning models due to limited data. Additionally, pictures of people are very similar to images available on ImageNet on which many sophisticated CNN models such as Inception, ResNet, and VGG have been trained . You can learn more about different CNN Architectures [here](https://medium.com/analytics-vidhya/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5). Due to limited GPU resources and project time constraints, I chose to use the VGG16 and ResNet50 models as my two base models. I unfroze the last 5 layers of VGG16 and ResNet50 for fine-tuning of model parameters. As mentioned earlier, you can find detailed code in the GitHub Repository.

To simplify the multi-classification task, I used a one-vs-rest multi-classification approach. Three binary classifiers are created for both VGG16 and ResNet50:
* eyewear vs no eyewear
* beard vs no beard
* hat versus no hat.

![Model Results](/assets/img/blog1_img3.png){:class="post-image img-blur"}.

The graph above shows the ROC curves and AUC values for all 6 of the best trained models. The top 3 curves are for VGG16 models and the bottom 3 curves are for the ResNet50 models. The labels are eyewear, beard, and hat from left to right. The AUC values for all of the models are 0.93 or higher. Surprisingly, VGG16 performed better than ResNet50 for all three classifiers. This could be due to limited dataset since ResNet50 has more layers than VGG16.
Based upon these 6 models, I made a prototype python program called product_recommender that can take in an image, folder or collection of folders and output pictures with a recommended product as image title. You can download the python file and test it out yourself [here](https://github.com/mzhou356/img2recommender/blob/master/python_files/product_recommender.py).

A few future plans for this project are:

Face Focused Concept:
* Play more with learning rate, learning rate decay, and momentum.
* Adjust regularization for ResNet50 such as dropout rate and batch normalization layers.
* Add more images especially with pictures that are falsely predicted.
* Go from transfer learning to full CNN with more images collected and labelled.
* Find better face recognition software for integration.

Recommender Concept:
* Develop complex CNN models as sample size increases exponentially.
* Expand products from hat, beard and eyewear items to jewelry, accessories, shoes, electronics, and more.
* Find various product identifier software for integration.

If you have any questions or suggestions for this project, please don't hesitate to contact me.
