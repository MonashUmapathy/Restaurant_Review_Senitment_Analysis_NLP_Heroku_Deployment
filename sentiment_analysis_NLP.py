# Importing Libaries

import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle


#Loading Dataset

#Dataset source: https://www.kaggle.com/vigneshwarsofficial/reviews/kernels


dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t')

# Cleaning the Dataset

corpus = []

ps = PorterStemmer()

for i in range(0,len(dataset)):
    
   # Cleaning special character from the reviews
  review = re.sub(pattern='[^a-zA-Z]',repl=' ', string=dataset['Review'][i])

  # Converting the entire review into lower case
  review = review.lower()

  # Tokenizing the review by words
  review_words = review.split()

  # Removing the stop words
  review_words = [word for word in review_words if not word in set(stopwords.words('english'))]

  # Stemming the words
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review_words]

  # Joining the stemmed words
  review = ' '.join(review)

  # Creating a corpus
  corpus.append(review)
  
# Creating the model using Bag of Words 

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Creating a pickle file for the CountVectorizer
pickle.dump(cv, open('cv-transform.pkl', 'wb'))


# Creating a model using a TF-IDF
     
###from sklearn.feature_extraction.text import TfidfVectorizer

####cv = TfidfVectorizer()

###X = cv.fit_transform(corpus).toarray()
###y = dataset.iloc[:, 1].values





# Model Building

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred) #[[55 42
                                     # 12 91]]
cr = classification_report(y_test,y_pred)

# 77% accuracy when using the MulinomialNB




# Fitting Naive Bayes to the Training set (GaussianNB)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred) #[[55 42
                                     # 12 91]]
cr = classification_report(y_test,y_pred)




# 73% accuracy when using the GaussianNB

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train,y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred) #[[55 42
                                     # 12 91]]
cr = classification_report(y_test,y_pred)


# 75% accuracy when using the GaussianNB


# Random Forest

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred) #[[55 42
                                     # 12 91]]
cr = classification_report(y_test,y_pred)

# 72% accuracy when using the GaussianNB

'''

    USUALLY IN NLP BEST SUIT MODELS ARE:
    Naives_bayes
    Decision Tree
    CART

 Performance by different Algorithm are same for both CountVectorizer and TfidfVectorizer :
     
     1.Naive Bayes(MultinomialNB) - 77%
     2.Naive Bayes(GaussianNB) - 73%
     3.Naive Bayes(BernoulliNB)-75%
     4.Random Forest -72%
     
     out of all, Naive Bayes(MultinomialNB) is at the TOP list. So we will choose Naive Bayes(MultinomialNB)
     
     GaussianNB:
         When dealing with continuous data, a typical assumption is that the continuous values associated with each class are distributed according to a normal (or Gaussian) distribution. 
     
     MultinomialNB:
         It is user for discrete data. In text classification, word occuring in the document we have the count how often occurs in doucment.
         
     BernoulliNB:
         It is used if your feacutre vectors are binary

'''
# Hyper Parameter tunning for MultinomialNB

from sklearn import metrics

previous_score=0
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    print("Alpha: {}, Score : {}".format(alpha,score))
    
    '''
    Alpha: 0.0, Score : 0.765
    Alpha: 0.1, Score : 0.78
    Alpha: 0.2, Score : 0.785
    Alpha: 0.30000000000000004, Score : 0.78
    Alpha: 0.4, Score : 0.78
    Alpha: 0.5, Score : 0.775
    Alpha: 0.6000000000000001, Score : 0.775
    Alpha: 0.7000000000000001, Score : 0.775
    Alpha: 0.8, Score : 0.77
    Alpha: 0.9, Score : 0.765

    
    '''
    
# Choosing the right Aplha value

classifier = MultinomialNB(alpha=0.2)
classifier.fit(X_train, y_train)


#Accuracy

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred) #[[55 42
                                     # 12 91]]
cr = classification_report(y_test,y_pred)


# After Hyper Tunning accuracy has been improved by 2% {Naive Bayes(MultinomialNB) - 79%}

# Creating a pickle file for the Multinomial Naive Bayes model
filename = 'restaurant-sentiment-mnb-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))



# To predict the unseen data

new_review = "The food was Awesome"

def predict(new_review):   

        new_review = re.sub("[^a-zA-Z]", " ", new_review)   

        new_review = new_review.lower().split()

        new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]   

        new_review = " ".join(new_review)   

        new_review = [new_review]   

        new_review = cv.transform(new_review).toarray()   

        if classifier.predict(new_review)[0] == 1:

            return "Positive"   

        else:       

            return "Negative"

       

feedback = predict(new_review)

print("This review is: ", feedback)

  

  


    
    
    
                     
                     
    
