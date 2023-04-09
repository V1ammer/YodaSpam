import os
import re
import nltk
import string
import joblib
import numpy as np
from email import *
import email.policy

import pandas as pd
from bs4 import BeautifulSoup
from collections import Counter
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def set_filenames():
    ham_filenames = [name for name in sorted(os.listdir('./hamnspam/ham')) if len(name) > 20]
    spam_filenames = [name for name in sorted(os.listdir('./hamnspam/spam')) if len(name) > 20]

    return ham_filenames, spam_filenames


def load_email(is_spam, filename):
    directory = "./hamnspam/spam" if is_spam else "./hamnspam/ham"
    with open(os.path.join(directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


def set_emails(ham_filenames, spam_filenames):
    ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
    spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]

    return ham_emails, spam_emails


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


def preprocess_data(text, stop_words):
    text = text.get_content()
    text = clean_text(text)
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)

    return text


def html_to_plain(email):
    try:
        soup = BeautifulSoup(email.get_content(), 'html.parser')
        return soup.text.replace('\n\n', '')
    except:
        return "empty"


def email_to_plain(email):
    struct = get_email_structure(email)
    for part in email.walk():
        partContentType = part.get_content_type()
        if partContentType not in ['text/plain', 'text/html']:
            continue
        try:
            partContent = part.get_content()
        except:
            partContent = str(part.get_payload())
        if partContentType == 'text/plain':
            return partContent
        else:
            return html_to_plain(part)


class EmailToWords(BaseEstimator, TransformerMixin):
    def __init__(self, stripHeaders=True, lowercaseConversion=True, punctuationRemoval=True,
                 urlReplacement=True, numberReplacement=True, stemming=True):
        self.stripHeaders = stripHeaders
        self.lowercaseConversion = lowercaseConversion
        self.punctuationRemoval = punctuationRemoval
        self.urlReplacement = urlReplacement
        self.numberReplacement = numberReplacement
        self.stemming = stemming
        self.stemmer = nltk.PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_to_words = []
        for email in X:
            text = email_to_plain(email)
            if text is None:
                text = 'empty'
            if self.lowercaseConversion:
                text = text.lower()

            if self.punctuationRemoval:
                text = text.replace('.', '')
                text = text.replace(',', '')
                text = text.replace('!', '')
                text = text.replace('?', '')

            word_counts = Counter(text.split())
            if self.stemming:
                stemmed_word_count = Counter()
                for word, count in word_counts.items():
                    stemmed_word = self.stemmer.stem(word)
                    stemmed_word_count[stemmed_word] += count
                word_counts = stemmed_word_count
            X_to_words.append(word_counts)
        return np.array(X_to_words)


class WordCountToVector(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_word_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_word_count[word] += min(count, 10)
        self.most_common = total_word_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(self.most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))


def create_pipeline():
    email_pipeline = Pipeline([
        ("Email to Words", EmailToWords()),
        ("Wordcount to Vector", WordCountToVector()),
    ])

    print("email_pipline: ", email_pipeline)

    return email_pipeline


def set_xy(ham_emails, spam_emails):
    X = np.array(ham_emails + spam_emails, message.EmailMessage)
    y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def preprocess_data_2(text, stop_words):
    text = clean_text(text)
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)

    return text


def main():
    stop_words = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']

    ham_filenames, spam_filenames = set_filenames()
    ham_emails, spam_emails = set_emails(ham_filenames, spam_filenames)

    email_pipeline = create_pipeline()

    X_train, X_test, y_train, y_test = set_xy(ham_emails, spam_emails)

    new_X_train = []
    new_X_test = []

    count = 0
    for item in X_train:
        try:
            new_X_train.append(preprocess_data(item, stop_words))
        except:
            new_X_train.append("")

    for item in X_test:
        try:
            new_X_test.append(preprocess_data(item, stop_words))
        except:
            new_X_test.append("")

    new_X_train = pd.Series(new_X_train)
    y_train = pd.Series(y_train)

    new_X_test = pd.Series(new_X_test)
    y_test = pd.Series(y_test)

    vect = CountVectorizer()
    vect.fit(new_X_train)
    X_train_dtm = vect.transform(new_X_train)
    X_test_dtm = vect.transform(new_X_test)

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_train_dtm)
    X_train_tfidf = tfidf_transformer.transform(X_train_dtm)
    X_test_tfidf = tfidf_transformer.transform(X_test_dtm)

    rf = RandomForestClassifier(max_depth=None, max_features='sqrt', min_samples_leaf=1, min_samples_split=10,
                                n_estimators=50)
    # rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_tfidf, y_train)

    print(type(X_train_tfidf), X_train_tfidf.shape)
    print(rf.score(X_train_tfidf, y_train))
    print(type(X_test_dtm), X_test_dtm.shape)
    print(rf.score(X_test_dtm, y_test))

    massages = [
        'Get rich quick!', 'Congratulations! You won a prize!', 'You win 10000$', 'suck sex',
        'Congratulations ur awarded $500 ', 'I cant pick the phone right now. Pls send a message',
        "i want sex", "free hot girls near you", "i love you", "hello, my friend",
        "i fuck your dog, because you won 5000 dollars"
    ]

    preprocess_data_2(massages, stop_words)
    massages_dtm = vect.transform(massages)
    massages_tfidf = tfidf_transformer.transform(massages_dtm)
    prediction = rf.predict(massages_tfidf)
    print(prediction)

    joblib.dump(rf, 'first_model.pkl')
    joblib.dump(vect, 'first_vect.pkl')
    joblib.dump(tfidf_transformer, 'first_tfidf_transformer.pkl')


if __name__ == '__main__':
    main()
