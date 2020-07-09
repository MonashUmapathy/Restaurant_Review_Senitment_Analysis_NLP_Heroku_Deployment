# Importing essential libraries
from flask import Flask, render_template, request,url_for
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()


# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'restaurant-sentiment-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))


app = Flask(__name__)

@app.route('/')
@app.route("/home")
def home():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])

def predict():
    if request.method == 'POST':
        message = request.form['message']
        new_review =  re.sub("[^a-zA-Z]", " ",message)
        new_review = new_review.lower().split()

        new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]

        new_review = " ".join(new_review)
        data = [new_review]
        vect = cv.transform(data).toarray()
        my_prediction =  classifier.predict(vect)
        return render_template('result.html', prediction= my_prediction)




if __name__ == '__main__':
	app.run()
