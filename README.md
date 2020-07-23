
# Restaurant_Review_Senitment_Analysis_NLP_Heroku_Deployment
---
![https://www.youtube.com/watch?v=bpdvNwvEeSE](https://img.shields.io/badge/Dataset-Kaggle-green.svg) ![Python 3.6](https://img.shields.io/badge/Python-3.6-Pink.svg) ![scikit-learnn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)

# Table of content
---
- Demo
- Overview
- Technical Aspect
- Installation
- Run
- Deployement on Heroku
- Directory Tree
- Bug / Feature Request
- Technologies Used

# Demo
---
### Link: (https://sentiment-analysis-nlp-deploy.herokuapp.com/)

# Overview

This is a senitment reviews classificiation Flask app trained on the top of Navie Baiyes API. The trained model (restaurant-sentiment-mnb-model.pkl) takes an text as an input and removing the common words as BOW and converting into matrix(0 and 1) by using count vectorizer and using a naive_bayes alogorthim predicting the review as positive or negative.

# Technical Aspect
---

This project is divided into two part:

1.Training a Machine learning model using Naive Bayes as MultinomialNB. Reason of using the MultinomialNB it is used for discrete data. In text classification, word occuring in  the document we have the count how often occurs in doucment. By Hypertuning the alpha parameter by using the grid search to find out the optimal alpha value.
 1. You can refer the code in sentiment_analysis_NLP.py.
    - Here i have tested other alogrithm like Naives_bayes, Decision Tree, CART, WordEmbedding out this Naives_bayer performed very well at the accuracy of 77% when comparing to other alogrithm
 
 2.Building and hosting a Flask web app on Heroku.
  1.User needs to give the review in the text area.
  2. Click predict button.
  3. Model will predict either a positive review or negative review along with gif.
  #### Note: sarcastic review is not predicted as well. Due to less data set accuracy level is 77%  
