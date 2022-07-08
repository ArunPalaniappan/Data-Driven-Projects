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

