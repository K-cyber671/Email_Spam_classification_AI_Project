# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 15:32:38 2026

@author: 1439182
"""

data = {
    'text': [
        'Get a free lottery prize now! Click here',
        'Hey, are we still meeting for lunch today?',
        'Urgent: Your bank account is compromised. Click to verify.',
        'Please review the attached project schedule.',
        'WINNER! You have won a cash prize of $500,000!',
        'Can you send me the notes from yesterday\'s lecture?'
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Ham
}

# Import the neccessary libraries

from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


Data = pd.DataFrame(data)

#Create training data and test data for model
X_train, X_test, y_train, y_test = train_test_split(Data["text"],Data["label"]
                                                    ,test_size = 0.4, random_state = 45)

#Define estimator model
Bayes = GaussianNB()

#Define Vectorizer for feature extraction
Vect = TfidfVectorizer()

#Extract features for model
X_train_vect = Vect.fit_transform(X_train).toarray()
X_test_vect = Vect.transform(X_test).toarray()

#Train the model
Bayes_fit = Bayes.fit(X_train_vect,y_train)

#Use the model to make the prediction
Bayes_pred = Bayes.predict(X_test_vect)

#display prediction
print(Bayes_pred)

#Check for accuracy via the classification report and accuracy score
print(classification_report(y_test,Bayes_pred))
print(accuracy_score(Bayes_pred,y_test))



