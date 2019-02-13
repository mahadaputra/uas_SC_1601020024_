# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 23:43:16 2019

@author: KarmaKum
"""

import numpy as np
import matplotlib.pyplot as plt      #projek UAS Soft Computing
import pandas as pd                  #I Kadek Mahada Putra
import keras                         #1601020024
from keras.models import Sequential  #Teknik Informatika
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:-1].values
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_namanegara_X = LabelEncoder()
labelencoder_jnskelamin_X = LabelEncoder()

X[:, 1] = labelencoder_namanegara_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_jnskelamin_X.fit_transform(X[:, 2])

onehenc = OneHotEncoder(categorical_features = [1])
X = onehenc.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 150)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)