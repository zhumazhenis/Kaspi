# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Task 1
# Vizualize data for preprocessing
dataset = pd.read_csv('kaspi1.csv')
aveCost = dataset.iloc[:,-2].values.astype(float)
y = dataset.iloc[:, -1].values
y = y / aveCost
listOfFeatures = ['FUEL_TYPE', 'BODY_TYPE', 'TYPE_OF_DRIVE','INTERIOR_TYPE', 'TRANSM_TYPE', 'AUTO_CONDITION']

for col in listOfFeatures:
    for s in dataset[col].unique():
        k = dataset.loc[dataset[col] == s].index.values
        sns.distplot(y[list(k)], kde = False, label = s)
        plt.legend()
        plt.ylabel('Count')
        plt.xlabel('Estimated_cost / Average_cost')
        plt.title(col)
    plt.show()
    plt.close()

year = dataset['YEAR'].values
for i in range(year.size):
    if year[i] < np.int64(5):
        year[i] = 0
    elif year[i] < np.int64(10):
        year[i] = 1
    elif year[i] < np.int64(15):
        year[i] = 2
    else: 
        year[i] = 3

for s in list(np.unique(year)):
    sns.distplot(y[year == s], kde = False, label = s.astype(str))
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel('Estimated_cost / Average_cost')
    plt.title('YEAR')


# Task 2
# Vizualize data for preprocessing
data = pd.read_csv('kaspi2.csv').head(100)
data = data.drop(['NUM','F120'], axis = 1)
data = data.fillna(data.mean())

y = data.TARGET
data = data.drop('TARGET', axis = 1)
dataCopy = data.copy()

sns.set(style="whitegrid", palette="muted")

features = [2,7,8,10,11,19,32,33,34,35,38,39,40,46,63,86,87,88, 77,78,79,96,108,109,119,120,125]

for i in features:
    dataCopy = pd.concat([y,data.iloc[:,i]],axis=1)
    dataCopy = pd.melt(dataCopy,id_vars="TARGET", var_name="features", value_name='value')
    plt.figure(figsize=(6,6))
    sns.swarmplot(x="features", y="value", hue="TARGET", data=dataCopy)
    plt.xticks(rotation=90)
