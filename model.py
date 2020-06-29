import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle

df = pd.read_csv("data.csv", encoding = "latin-1")
x = df["message"]
y = df["class"].values
y = np.where(y=="ham", 0, 1)

cv = CountVectorizer()
x = cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(x_train,y_train)
pickle.dump(clf, open("trained.pkl", 'wb'))