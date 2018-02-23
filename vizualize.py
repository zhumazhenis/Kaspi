# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
