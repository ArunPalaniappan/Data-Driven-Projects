import pandas as pd
import numpy as np
import datetime
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

uber_data = pd.read_csv('Kaggle-uber.csv')

hours = uber_data['HOUR'].value_counts()
hours.plot(kind='bar', color='red', figsize=(10,5))
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.title('Number of trips Vs hours')

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

data['id']= label_encoder.fit_transform(data['id'])
data['datetime']= label_encoder.fit_transform(data['datetime']) 3 data['timezone'] = label_encoder.fit_transform(data['timezone'])
data['destination']= label_encoder.fit_transform(data['destination'])
data['product_id']= label_encoder.fit_transform(data['product_id'])
data['short_summary']= label_encoder.fit_transform(data['short_summary'])
data['long_summary']= label_encoder.fit_transform(data['long_summary'])
data['name']= label_encoder.fit_transform(data['name'])

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.feature_selection import RFE
from sklearn. ensemble import RandomForestRegressor
from sklearn import ensemble

def train_test_models(X_train, y_train, X_test, y_test):
  print("Random Forest...")
  random_forest = RandomForestRegressor(n_estimators = 20, random_state = 0)
  random_forest.fit(X_train, y_train)
  print(random_forest.score(X_test, y_test))
  
  return [random_forest]

