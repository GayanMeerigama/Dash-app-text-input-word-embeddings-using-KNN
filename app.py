from flask import Flask, render_template, request
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from scipy import sparse
import pickle 



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/essay" , methods=['GET', 'POST'])
def essay():
        #value = ['Modern terrorism is attracted to the media, and some extreme terrorist groups use it since it is the role of the media to report on any significant event. Moreover, extreme terrorist acts use the media since spectacular and dramatic terrorism aspects fascinate the public. However, terrorism should not impact the']
        #value = ['Technology has revolutionized various aspects of our lives, and one domain that has witnessed significant transformation is education. The integration of technology in educational settings has redefined the learning experience, enhancing access to information, fostering interactive engagement, and promoting personalized instruction. In this essay, we will explore the profound impact of technology on education, including its role in expanding educational opportunities, improving teaching methods, and preparing students for the demands of the 21st century']

        # load the vectorizer
        value = request.form['text']
        value=[value]
        
        #Load it later
        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
        #count_vect = CountVectorizer()
        X_new_counts = loaded_vec.transform(value)

        #tfidf_transformer = TfidfTransformer()
        loaded_tfidf = pickle.load(open("tfidf1.pkl", 'rb'))
        X_new_tfidf = loaded_tfidf.transform(X_new_counts)

        #loaded_model 
        loaded_model = pickle.load(open('lassifier_model.pkl', 'rb'))

        # make a prediction
        prediction=loaded_model.predict(X_new_tfidf)

        return render_template('index.html', prediction_text=prediction)

if __name__ == '__main__':
    app.run(debug=True)
