from flask import Flask,render_template,redirect,request,jsonify
import pickle
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

# downloading the list of stopword
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
app=Flask(__name__)


loaded_model = pickle.load(open('./static/model/knn_model_senti1', 'rb'))
vecto=pickle.load(open('./static/model/vecto1.pkl', 'rb'))

@app.route("/",methods=["GET"])
def home():
    # if request.method=='GET':
    #     render_template('home.html')
    return render_template("home.html")

@app.route("/predict",methods=["POST","GET"])
def predict():
    print("Hello")
    if request.method=='POST':
        print("kello")
        review=request.form['review']
        print(review)
        prediction=predict_sentiment(review)
        print(prediction)
        data={
            "prediction":prediction[0]
        }
        return jsonify(data)
    return redirect("/")

def predict_sentiment(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = vecto.transform([final_review]).toarray()
  return loaded_model.predict(temp)

if __name__ == "__main__":
    app.run()