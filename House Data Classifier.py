# program for predicting the house values based on its features
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# getting the data from CSV file
data = pd.read_csv('california_housing_train.csv')
x = data['total_rooms'].values
y = data['median_house_value'].values
# getting names of the features
feature_names = data.keys()
feature_names = feature_names[:8]
# putting features in a separate dataframe
features = data[feature_names].values
y = np.array(y).reshape(-1, 1)

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.decomposition import PCA

clf = linear_model.LinearRegression()
# by using Principal Component Analysis we decompose 9 features to 1
pca = PCA(n_components = 1)
x_train, x_test, y_train, y_test = train_test_split(features, y)
pca.fit(x_train)
# changing the dimensions of x_train, x_test
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
from sklearn.metrics import r2_score
score = r2_score(y_test, pred)
print('Accuracy score: ', score)

# plotting the values
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_test, pred, color = 'orange')
plt.show()


