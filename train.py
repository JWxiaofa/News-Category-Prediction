from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)
import numpy as np
import train_data
import dev_data
import test_data


y_train = np.array([row['category'] for row in train_data.data])
y_dev = np.array([row['category'] for row in dev_data.data])
y_test = np.array([row['category'] for row in test_data.data])

vectorizers = [CountVectorizer(), CountVectorizer(ngram_range=(2, 2)), TfidfVectorizer()]
models = [MultinomialNB(), LogisticRegression(), DecisionTreeClassifier()]

for vectorizer in vectorizers:
    X_train = vectorizer.fit_transform([row['text'] for row in train_data.data])
    X_dev = vectorizer.transform([row['text'] for row in dev_data.data])
    for model in models:
        model.fit(X_train, y_train)
        y_pred_dev = model.predict(X_dev)
        accuracy = accuracy_score(y_dev, y_pred_dev)
        print(vectorizer, model, accuracy)


parameters = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

# unigram & NB
print("unigram & NB")
X_train = vectorizers[0].fit_transform([row['text'] for row in train_data.data])
X_dev = vectorizers[0].transform([row['text'] for row in dev_data.data])
for parameter in parameters:
    model = MultinomialNB(alpha=parameter)
    model.fit(X_train, y_train)
    y_pred_dev = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred_dev)
    print(parameter, accuracy)

# unigram & Logistic Regression
print("unigram & LR")
X_train = vectorizers[0].fit_transform([row['text'] for row in train_data.data])
X_dev = vectorizers[0].transform([row['text'] for row in dev_data.data])
for parameter in parameters:
    model = LogisticRegression(C=parameter)
    model.fit(X_train, y_train)
    y_pred_dev = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred_dev)
    print(parameter, accuracy)

# tfidf & LR
print("tfidf & LR")
X_train = vectorizers[2].fit_transform([row['text'] for row in train_data.data])
X_dev = vectorizers[2].transform([row['text'] for row in dev_data.data])
for parameter in parameters:
    model = LogisticRegression(C=parameter)
    model.fit(X_train, y_train)
    y_pred_dev = model.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred_dev)
    print(parameter, accuracy)

# configuration 1
print("unigram NB k=0.5")
X_train = vectorizers[0].fit_transform([row['text'] for row in train_data.data])
X_test = vectorizers[0].transform([row['text'] for row in test_data.data])
model = MultinomialNB(alpha=0.5)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(accuracy)

# configuration 2
print("unigram LR C=0.1")
X_train = vectorizers[0].fit_transform([row['text'] for row in train_data.data])
X_test = vectorizers[0].transform([row['text'] for row in test_data.data])
model = LogisticRegression(C=0.1)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(accuracy)

# configuration 3
print("tdidf LR C=1.0")
X_train = vectorizers[2].fit_transform([row['text'] for row in train_data.data])
X_test = vectorizers[2].transform([row['text'] for row in test_data.data])
model = LogisticRegression(C=1.0)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)
print(accuracy)
print(classification_report(y_test, y_pred_test))



