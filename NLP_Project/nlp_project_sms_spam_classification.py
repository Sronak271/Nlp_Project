# -*- coding: utf-8 -*-
"""NLP_Project_SMS Spam Classification.ipynb

# SMS Spam Classification

Natural Language Processing Using Machine Learning
"""

# Loading the libraries
import pandas as pd
import numpy as np
import nltk 
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Reading the csv file
messages = pd.read_csv("sms_spam.txt", sep='\t', names=['Label', 'Message'])

# Specifying the names of the columns while reading csv file (tsv--tab separated values)

messages.head()

"""## Exploratory Data Analysis"""

# Info about the data
messages.info()

# Finding missing values
messages.isnull().sum()

# Shape of the dataframe
messages.shape

# Target variables counts
messages['Label'].value_counts()

# Data is imbalanced but for now we will continue with this

"""## Data Preprocessing

**Calculating length of message**
"""

# Calculating length of message
mes_len = 0
length = []
for i in range(len(messages)):
    length.append(len(messages['Message'][i]))

length

# Adding Length column to the dataframe
messages['Length'] = length

messages.head()

"""**Calculating Punctuations in each message**"""

# Calculating Punctuations in each message

import string
count = 0
punct=[]
for i in range(len(messages)):
    for j in messages['Message'][i]:
        if j in string.punctuation:
            count += 1
    #print(count)
    punct.append(count)
    count = 0

punct

# Adding punctuation length column to dataframe
messages["Punctuation"]=punct

"""<br>

### Text Cleaning
"""

# Regex
import re

# Stopwords
from nltk.corpus import stopwords

# Lemmatization
from nltk.stem import WordNetLemmatizer
# Creating object for Lemmatizer
lemmatizer = WordNetLemmatizer()

# Removal of extra characters and stop words and lemmatization
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
corpus = []

# Skipping the 0th index (it's of Label)
for i in range(0,len(messages)):
    words = re.sub('[^a-zA-Z]', ' ', messages['Message'][i])
    words = words.lower()
    # Splits into list of words
    words = words.split()
    
    # Lemmatizing the word and removing the stopwords
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    
    # Again join words to form sentences
    words = ' '.join(words)
    
    corpus.append(words)

# What's in Corpus
corpus[0]

# Replacing Original Message with the Transformed Messages
messages['Message'] = corpus

messages.head()

"""<br>

### Analyzing the difference between Spam and Ham messages
"""

spam_messages = messages[messages['Label'] == 'spam']
ham_messages = messages[messages['Label'] == 'ham']

spam_messages.head()

ham_messages.head()

spam_messages['Length'].mean()

ham_messages['Length'].mean()

"""We can see that Spam messages have more average words than Ham messages"""

spam_messages['Punctuation'].mean()

ham_messages['Punctuation'].mean()

"""Same with Punctuation also, We can see that Spam messages have more average punctuation than Ham messages

## Model Building
"""

X = messages['Message']

X.head()

y = messages['Label']

y.head()

"""### Train Test Split"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train.head()

"""### Demonstration of Count Vectorizer

(Bag of Words)
"""

from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()

X_train_count_vect = count_vect.fit_transform(X_train).toarray()

X_train_count_vect

# 670 are the sentences and 2087 are the words in total sentences
X_train_count_vect.shape

"""**Note:-**<br>
There might be that, some words in 5772 words are not frequently present and are just appearing 1-2 times, we can reduce them using cv = CountVectorizer(max_features = 4000) (an approach)

This will only take 4000 words leading to coming of most frequent words

    We can change the max_features, according to what we want

### Demonstration of TF-IDF Vectorizer

(Term Frequency - Inverse Document Frequency)


CountVectorizer(Bag of Words) + TFIDF Transformer, Scikit-Learn has provided with a method of TFIDF vectorizer (combining two steps into one)
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X_train_tfidf_vect = count_vect.fit_transform(X_train).toarray()

X_train_tfidf_vect

X_train_tfidf_vect.shape

"""## Pipelining

We are doing pipelining as we need to perform the same procedures for the test data to get predictions, that may be tiresome.

However what convenient about this pipeline object is that it actually can perform all these steps for you in a single cell, that means you can directly provide the data and it will be both vectorized and run the classifier on it in a single step.

Pipeline takes list of tuple.
"""

from sklearn.pipeline import Pipeline

"""### Naive Bayer Classifier"""

from sklearn.naive_bayes import MultinomialNB

# each tuple takes the name you decide , next you call what you want to occur
text_mnb = Pipeline([('tfidf', TfidfVectorizer()), ('mnb', MultinomialNB())])

# Now u can directly pass the X_train dataset.
text_mnb.fit(X_train, y_train)

X_test.head()

# It will take the X_test and do all the steps, vectorize it and predict it
y_preds_mnb = text_mnb.predict(X_test)

# Predictions of the test data
y_preds_mnb

# Training score
text_mnb.score(X_train, y_train)

# Testing score
text_mnb.score(X_test, y_test)

"""**Evaluation Metrics**"""

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_preds_mnb))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_preds_mnb))

"""### SVM Classifier"""

from sklearn.svm import LinearSVC

# each tuple takes the name you decide , next you call what you want to occur
text_svm = Pipeline([('tfidf', TfidfVectorizer()), ('svm', LinearSVC())])

# Now u can directly pass the X_train dataset.
text_svm.fit(X_train, y_train)

X_test.head()

# It will take the X_test and do all the steps, vectorize it and predict it
y_preds_svm = text_svm.predict(X_test)

# Predictions of the test data
y_preds_svm

# Training score
text_svm.score(X_train, y_train)

# Testing score
text_svm.score(X_test, y_test)

"""**Evaluation Metrics**"""

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_preds_svm))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_preds_svm))

"""### Prediciting on New SMS"""

from tkinter import *


class MyWindow:
    def __init__(self, win):
        self.lbl1 = Label(win, text='Enter the Message', font="Times")
        self.lbl2 = Label(win, text='Result', font="Times")
        self.t1 = Text(win, height=3, width=60, wrap=WORD)
        self.t2 = Entry()
        self.lbl1.place(x=100, y=50)
        self.t1.place(x=100, y=80)
        self.b1 = Button(win, text='Check', font="Times", command=self.spam)
        self.b1.place(x=100, y=150)
        self.lbl2.place(x=100, y=200)
        self.t2.place(x=100, y=240)

    def spam(self):
        text = self.t1.get("1.0", 'end-1c')

        def refined_text(text):
            # Removal of extra characters and stop words
            words = re.sub('[^a-zA-Z]', ' ', text)
            words = words.lower()
            # Splits into list of words
            words = words.split()

            # Lemmatizing the word and removing the stopwords
            words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]

            # Again join words to form sentences
            words = ' '.join(words)
            return words

        refined_word = refined_text(text)
        refined_word = [refined_word]
        refined_word

        # Directly predicting the single message to the model
        result_arr = text_mnb.predict(refined_word)
        result = ' '.join(map(str, result_arr))
        self.t2.delete(0, 'end')
        self.t2.insert(END, str(result))


window = Tk()
mywin = MyWindow(window)
window.title('SMS Spam Detector')
window.geometry("650x450+10+10")
Label(window, text="SMS Spam Detector", fg="red", font="Times 20").pack()
window.mainloop()

