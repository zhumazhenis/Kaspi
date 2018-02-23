# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('kaspi1.csv')

# YEAR, AUTO_CONDITION, AVG_COST features are selected 
X = dataset.iloc[:, [1,-3]].values
aveCost = dataset.iloc[:, 15].values
y = dataset.iloc[:, -1].values

# Convertign years to categorical data
year = X[:, [0]]
for i in range(year.size):
    if year[i] < np.int64(5):
        year[i] = 0
    elif year[i] < np.int64(10):
        year[i] = 1
    elif year[i] < np.int64(15):
        year[i] = 2
    else: 
        year[i] = 3    
        
encoder = OneHotEncoder(categorical_features=[0])
encoder.fit(year)
year = encoder.transform(year).toarray()
year = year[:, 1:]

# Car condition categorical data
labelEncoder = LabelEncoder()
X[:, 1] = labelEncoder.fit_transform(X[:, 1])
encoder = OneHotEncoder(categorical_features=[1])
encoder.fit(X)
X = encoder.transform(X).toarray()
X = X[:, 1:-1]

# Ready for processing
X = np.append(year, X, axis = 1)
one = np.ones((X[:,0].size,1))
X = np.append(one, X, axis = 1)
for i in range(6):
    X[:,i] = X[:,i] * aveCost

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Logistic Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.coef_

# Predicting the Train set results
y_pred = regressor.predict(X_train)
# Error percentage
error_train = abs(y_pred / y_train * 100 - 100)
# n is the number of predictions in which error < 10%
temp = error_train < 10
n_train = list(temp).count(True)
# 'accuracy' is the analog of 78% that you have mentioned in the instruction
accuracy_train = n_train / y_train.size * 100

# Predicting the Test set results
y_pred = regressor.predict(X_test)
# Error percentage
error_test = abs(y_pred / y_test * 100 - 100)
# n is the number of predictions in which error > 10%
temp = error_test < 10
n_test = list(temp).count(True)
# 'accuracy' is the analog of 78% that you have mentioned in the instruction
accuracy_test = n_test / y_test.size * 100
