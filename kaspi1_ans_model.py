# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataTrain = pd.read_csv('kaspi1_train.csv')
dataTest = pd.read_csv('kaspi1_test.csv')

# YEAR, AUTO_CONDITION, AVG_COST features are selected 
# X = dataTrain.iloc[:, [1,-3]].values
# aveCost = dataTrain.iloc[:, 15].values
# y = dataTrain.iloc[:, -1].values

X_train = dataTrain.iloc[:, [1,-3]].values
aveCost_train = dataTrain.iloc[:, -2].values
y_train = dataTrain.iloc[:, -1].values

X_test = dataTest.iloc[:, [1,-2]].values
aveCost_test = dataTest.iloc[:, -1].values
id_test = dataTest.iloc[:, 0].values

# Convertign years to categorical data
year_train = X_train[:, [0]]
for i in range(year_train.size):
    if year_train[i] < np.int64(5):
        year_train[i] = 0
    elif year_train[i] < np.int64(10):
        year_train[i] = 1
    elif year_train[i] < np.int64(15):
        year_train[i] = 2
    else: 
        year_train[i] = 3  
        
year_test = X_test[:, [0]]
for i in range(year_test.size):
    if year_test[i] < np.int64(5):
        year_test[i] = 0
    elif year_test[i] < np.int64(10):
        year_test[i] = 1
    elif year_test[i] < np.int64(15):
        year_test[i] = 2
    else: 
        year_test[i] = 3

encoder = OneHotEncoder(categorical_features=[0])
encoder.fit(year_train)
year_train = encoder.transform(year_train).toarray()
year_train = year_train[:, 1:]
year_test = encoder.transform(year_test).toarray()
year_test = year_test[:, 1:]


# Car condition categorical data
labelEncoder = LabelEncoder().fit(X_train[:, 1])
X_train[:, 1] = labelEncoder.transform(X_train[:, 1])
X_test[:, 1] = labelEncoder.transform(X_test[:, 1])

encoder = OneHotEncoder(categorical_features=[1])
encoder.fit(X_train)
X_train = encoder.transform(X_train).toarray()
X_train = X_train[:, 1:-1]
X_test = encoder.transform(X_test).toarray()
X_test = X_test[:, 1:-1]


# Ready for processing
X_train = np.append(year_train, X_train, axis = 1)
one = np.ones((X_train[:,0].size,1))
X_train = np.append(one, X_train, axis = 1)
for i in range(6):
    X_train[:,i] = X_train[:,i] * aveCost_train

X_test = np.append(year_test, X_test, axis = 1)
one = np.ones((X_test[:,0].size,1))
X_test = np.append(one, X_test, axis = 1)
for i in range(6):
    X_test[:,i] = X_test[:,i] * aveCost_test
# Splitting the dataset into the Training set and Test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

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

# Predicting the Train set results
y_test = regressor.predict(X_test)

raw_data = {'ID': id_test, 'ESTIM_COST': y_test}
outFile = pd.DataFrame(raw_data, columns = ['ID', 'ESTIM_COST'])
outFile.to_csv('ans1.csv')