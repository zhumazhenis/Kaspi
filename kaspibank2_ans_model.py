
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from subprocess import check_output

# Importing the dataset
dataTrain = pd.read_csv('kaspi2_train.csv')
nulCheck = dataTrain.isnull().sum()
dataTrain = dataTrain.fillna(dataTrain.mean())
dataTrain = dataTrain.drop(['NUM','F120'], axis = 1)

dataTest = pd.read_csv('kaspi2_test.csv')
num = dataTest['NUM'].values
dataTest = dataTest.fillna(dataTest.mean())
dataTest = dataTest.drop(['NUM','F120'], axis = 1)

# Taking features
features = [2,7,8,10,11,19,32,33,34,35,38,39,40,46,63,86,87,88, 77,78,79,96,108,109,119,120,125]
X_train = dataTrain.iloc[:, features].values  
y_train = dataTrain.iloc[:, -1].values

X_test = dataTest.iloc[:, features].values  

# Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)

raw_data = {'NUM': num, 'TARGET': y_pred}
outFile = pd.DataFrame(raw_data, columns = ['NUM', 'TARGET'])
outFile.to_csv('ans2.csv')
