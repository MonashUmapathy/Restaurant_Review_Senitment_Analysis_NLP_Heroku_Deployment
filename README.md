
# Restaurant_Review_Senitment_Analysis_NLP_Heroku_Deployment

![https://www.youtube.com/watch?v=bpdvNwvEeSE](https://img.shields.io/badge/Dataset-Kaggle-green.svg) ![Python 3.6](https://img.shields.io/badge/Python-3.6-Pink.svg) ![scikit-learnn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)

# Table of content

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

### Link: (https://sentiment-analysis-nlp-deploy.herokuapp.com/)

# Overview

This is a senitment reviews classificiation Flask app trained on the top of Navie Baiyes API. The trained model (restaurant-sentiment-mnb-model.pkl) takes an text as an input and removing the common words as BOW and converting into matrix(0 and 1) by using count vectorizer and using a naive_bayes alogorthim predicting the review as positive or negative.

# Technical Aspect


This project is divided into two part:

1.Training a Machine learning model using Naive Bayes as MultinomialNB. Reason of using the MultinomialNB it is used for discrete data. In text classification, word occuring in  the document we have the count how often occurs in doucment. By Hypertuning the alpha parameter by using the grid search to find out the optimal alpha value.
 1. You can refer the code in sentiment_analysis_NLP.py.
    - Here i have tested other alogrithm like Naives_bayes, Decision Tree, CART, WordEmbedding out this Naives_bayer performed very well at the accuracy of 77% when comparing to other alogrithm
 
 2.Building and hosting a Flask web app on Heroku.
  1.User needs to give the review in the text area.
  2. Click predict button.
  3. Model will predict either a positive review or negative review along with gif.
  #### Note: sarcastic review is not predicted as well. Due to less data set accuracy level is 77%  
  
  # Installation
  
  The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
 
 ```
 pip install -r requirements.txt
 ```
 
 # Run
 
 ### Step1(Windows users)
 
 1.Download the files from repository and Install all the requirments.
 
 ### Step2
 1. Open termial
 
```
  cd"Path of the download file directory"
  python app.py 
```
 # Deployement on Heroku
 
 Our next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.
 
 # Directory Tree
 
 ```
 |--static
 |  |--css
 |     |--styles.css
 |  |--icon
 |     |--chef.png
 |--templates
 |--app.py
 |--cv-transform.pkl
 |--requirements.txt
 |--Restaurant_Reviews.tsv
 |--restaurant-sentiment-mnb-model.pkl
 
 ```
 # Bug / Feature Request
 If you find a bug (the website couldn't handle the query and / or gave undesired results), kindly open an issue [here]() by including your search query and the expected result.

If you'd like to request a new function, feel free to do so by opening an issue [here](). Please include sample queries and their corresponding results.

# Technologies Used

![https://www.youtube.com/watch?v=bpdvNwvEeSE](https://img.shields.io/badge/Dataset-Kaggle-green.svg) ![Python 3.6](https://img.shields.io/badge/Python-3.6-Pink.svg) ![scikit-learnn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)


![flask](https://user-images.githubusercontent.com/52874412/88377965-bc4e9100-cdbd-11ea-8a9e-ac225a8162f6.png "https://flask.palletsprojects.com/en/1.1.x/")

![heroku-logo](https://user-images.githubusercontent.com/52874412/88378611-f8cebc80-cdbe-11ea-9335-2c8e8d0a36a3.png)

![gunicorn](https://user-images.githubusercontent.com/52874412/88378615-fb311680-cdbe-11ea-817b-084773f70a68.jpg)

 
