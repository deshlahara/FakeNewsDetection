#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from textblob import TextBlob
from profanity_check import predict, predict_prob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.externals import joblib
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import matplotlib.pyplot as plt
from sklearn import metrics

app = Flask(__name__)


def FeatureSet(frame):
    global_set = []
    for row in frame:
        text = row
        profanity = predict_prob([text])[0]
        sentiment = TextBlob(text)
        polarity = sentiment.polarity
        subjectivity = sentiment.subjectivity
        # return [polarity, subjectivity, profanity]
        global_set.append([polarity, subjectivity, profanity])
    return global_set


def SarcasticSet(frame):
    global_set = []
    for row in frame:
        text = row
        profanity = predict_prob([text])[0]
        sentiment = TextBlob(text)
        polarity = sentiment.polarity
        # return [polarity, subjectivity, profanity]
        global_set.append([polarity, profanity])
    return global_set


def sarcasticFeature(text):
    profanity = predict_prob([text])[0]
    sentiment = TextBlob(text)
    polarity = sentiment.polarity
    return [profanity, polarity]


with open('tfidf_title.pkl', 'rb') as f:
    tfidf_title = pickle.load(f)

with open('passive_title.pkl', 'rb') as f:
    passive_title = pickle.load(f)

with open('svc_title.pkl', 'rb') as f:
    svc_title = pickle.load(f)


def calltext(text, tfidf_title, passive_title):
    # text = "hello this text is about autheticating news"

    title = np.array(FeatureSet([text]))

    transform = tfidf_title.transform([text])

    tfidf = transform.todense()

    pred = np.column_stack((tfidf, title))

    predict_text = passive_title.predict(pred)
    return predict_text


with open('tfidf_main.pkl', 'rb') as f:
    tfidf_main = pickle.load(f)

with open('passive.pkl', 'rb') as f:
    passive = pickle.load(f)


def calltitle(text, tfidf_main, passive):
    # text = " This text is fake "
    transform_main = tfidf_main.transform([text])

    tfidf_main = transform_main.todense()

    title_main = np.array(FeatureSet([text]))

    pred_main = np.column_stack((tfidf_main, title_main))

    predict_title = passive.predict(pred_main)

    return predict_title


with open('tfidf_bias.pkl', 'rb') as f:
    tfidf_bias = pickle.load(f)

with open('passive_bias.pkl', 'rb') as f:
    passive_bias = pickle.load(f)

with open('svc_bias.pkl', 'rb') as f:
    svc_bias = pickle.load(f)


def callbias(bias_text, bias_tfidf, passive_bias):
    bias_transform = tfidf_bias.transform([bias_text])
    bias_output = passive_bias.predict(bias_transform)
    return bias_output


with open('tfidf_sarcasm.pkl', 'rb') as f:
    tfidf_sarcasm = pickle.load(f)

with open('passive_sarcasm.pkl', 'rb') as f:
    passive_sarcasm = pickle.load(f)


def sarcasm(tfidf_sarcasm, passive_sarcasm, text):
    # text = "hello this text is about autheticating news"

    title = np.array(SarcasticSet([text]))

    transform = tfidf_sarcasm.transform([text])

    tfidf = transform.todense()

    pred = np.column_stack((tfidf, title))

    output = passive_sarcasm.predict(pred)
    return output


def predictoutput(text, title):
    a = calltitle(title, tfidf_title, passive_title)
    b = calltext(text, tfidf_main, passive)
    c = sarcasm(tfidf_sarcasm, passive_sarcasm, text)
    d = callbias(text, tfidf_bias, passive_bias)
    return [a[0], b[0], c[0], d[0]]


@app.route('/', methods=['GET', 'POST'])
def res():
    if request.method == 'POST':
        title = request.form.get('headline')
        text = request.form.get('comment')

        result = predictoutput(text, title)

        print(result)

        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction='')


if __name__ == '__main__':
    app.run(port=5000, debug=True)

    # text = request.form.to_dict(['comment'])
    # title = request.form.to_dict(['headline'])
    # x_text = list(x_text.values())
    # x_text = list(map(int, x_text))
    # x_title = list(x_title.values())
    # x_title = list(map(int, x_title))
    # x_sarcasm = list(x_sarcasm.values())
    # x_sarcasm = list(map(int, x_sarcasm))
    # text = list(text.values())
    # text = list(map(int, text))
    # title = list(title.values())
    # title = list(map(int, title))
    # result = predictoutput(text,title)

    # if result == [1,1,1,0]:
    #     prediction='Title: Fake Text: Fake, Sarcasm : True'
    # elif result == [0,0,0,0]  :
    #     prediction = 'Title: Real Text: Real, Sarcasm: False'

    # if result[0] == 1:
    #     prediction = 'Given text is fake'
    # else:
    #     prediction = 'Given text is real'

    # if result[1] == 1:
    #     prediction = 'Given headline is fake'
    # else:
    #     prediction = 'Given headline is real'

    # if result[2] == 1:
    #     prediction = 'Given content is sarcastic'
    # else:
    #     prediction = 'Given content is not sarcastic'

    # return render_template("predict.html", prediction=prediction)


