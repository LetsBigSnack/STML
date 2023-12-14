from flask import Flask, render_template, request, redirect, url_for

# Basic Data Handling and Visualization Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# NLP Libraries
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Machine Learning Libraries
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline, make_pipeline

# LIME and SHAP for Explainable AI
import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer
import shap

app = Flask(__name__, static_url_path='/static')

## Text - Cleaning

def RemoveSpecialCharacters(sentence):
    return re.sub('[^a-zA-Z0-9?!]+',' ',sentence)

def ConvertToLowerCase(sentence):
    return sentence.lower()

def ConvertAndRemove(sentence):
    sentence = str(sentence)
    sentence = RemoveSpecialCharacters(sentence)
    # convert to lower case
    sentence = ConvertToLowerCase(sentence)
    return sentence

def CleanText(sentence):
    sentence = str(sentence)

    # Remove stopwords
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', 'im', 'dont', 'doin', 'ure']
    # Remove punctuation
    nopunc = [char for char in sentence if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    sentence = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    sentence = ConvertAndRemove(sentence)
    return sentence



# Classifieres
Classifiers = [
{
   'label': 'Logistic_Regression',
   'model': LogisticRegression(C=0.00000001,solver='liblinear',max_iter=200, multi_class='auto'),
},
{
    'label': 'SGD_Classifier',
    'model': SGDClassifier(loss='log', warm_start=True, max_iter=1000, l1_ratio=0.03, penalty='l2', alpha=1e-4, fit_intercept=False),
},
{
    'label': 'KNeighbours',
    'model': KNeighborsClassifier(n_neighbors=15),
},
{
    'label': 'Decision_Tree',
    'model': DecisionTreeClassifier(max_depth=10,random_state=101,max_features= None,min_samples_leaf=15),
},
{
   'label': 'Random_Forest',
   'model': RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,random_state=101,max_features= None,min_samples_leaf = 30),
}
]


# Loads the Dataset from a local .csv file
df=pd.read_csv('data/clickbait_data.csv')

#Removing Special Characters and transforming text to lower case in the headline column
df['Text_cleaning'] = df.headline.apply(CleanText)

#Define the features (X) and label (Y)
X = df.Text_cleaning
y = df.clickbait

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_clean = tfidf_transformer.fit_transform(X_train_dtm)

pipe = Pipeline([('bow', CountVectorizer()),
                 ('tfid', TfidfTransformer()),
                 ('model', Classifiers[1]['model'])])
pipe.fit(X_train, y_train)

explainer = LimeTextExplainer(class_names=['Not Clickbait', 'Clickbait'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        text_input = request.form['text_input']
        prediction = pipe.predict([CleanText(text_input)])
        result = ""
        if prediction == 0:
            result = 'Not Clickbait'
        else:
            result = 'Clickbait'

        exp = explainer.explain_instance(CleanText(text_input), pipe.predict_proba)
        exp.show_in_notebook(text=True)
        exp.save_to_file('static/lime.html')

        return render_template('result.html', text_input=text_input, result = result, proba = str(pipe.predict_proba([CleanText(text_input)])))

if __name__ == '__main__':
    app.run(debug=True)
