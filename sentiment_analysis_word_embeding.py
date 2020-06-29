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
  
### Vocabulary size
voc_size=5000

#One Hot Representation

from keras.preprocessing.text import one_hot

onehot_repr=[one_hot(words,voc_size)for words in corpus] 
print(onehot_repr) 


from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense



sent_length=[]

for lst in range(0,len(onehot_repr)):
    length = len(onehot_repr[lst])
    sent_length.append(length)

sent_length=30

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())



import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(dataset.iloc[:, 1].values)



X_final.shape,y_final.shape


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)


### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

# Hyperparamter Tunning

from keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(dataset.iloc[:, 1].values)



X_final.shape,y_final.shape


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.1, random_state=42)


### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)



y_pred=model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix,classification_report

cm = confusion_matrix(y_test,y_pred) 

cr = classification_report(y_test,y_pred)


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

'''
 Conculsion:
     
     In this problem statement, Navie Baiyes is performing well with 79% accuracy is
     greater than wordembeding



'''













  