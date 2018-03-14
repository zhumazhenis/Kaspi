
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
data = pd.read_csv('kaspi2.csv')
data = data.fillna(data.mean())
data = data.drop(['NUM','F120'], axis = 1)

# Taking features

X = data.iloc[:, [2,7,8,10,11,19,32,33,34,35,38,39,40,46,63,86,87,88, 77,78,79,96,108,109,119,120,125]].values  
y = data.iloc[:, -1].values

# Splittin dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
print(classifier.feature_importances_)

# Prediction
y_pred = classifier.predict(X_train)
check = (y_pred == y_train)
n = list(check).count(True)
