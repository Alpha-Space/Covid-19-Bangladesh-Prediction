import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator

df = pd.read_csv('F:/time_series_covid19_confirmed_global.csv")

start = datetime.datetime.strptime("08-01-20", "%m-%d-%y")
end  = datetime.datetime.strptime("12-01-20", "%m-%d-%y")

date = start
date_list =[]
final_prediction = {}
date_comparison =[]

while(date.timestamp()<=end.timestamp()):
  date+=datetime.timedelta(days=1)
  date_comparison.append(date)
  date_list.append(date.timestamp())

x = df.columns[4:]
y = df[df["Country/Region"] == "Bangladesh"].drop(df.columns[:4], axis = 1)


#Function to change all the x datetimes into timestamps:
def swap(x, format="%m/%d/%y"):
    return datetime.datetime.strptime(x, format).timestamp()
x_stmps= pd.Series(x).apply(swap)




#Initializing the period to be predicted:
start = datetime.datetime.strptime("09-09-20", "%m-%d-%y")
end = datetime.datetime.strptime("12-07-20", "%m-%d-%y")
date = start
date_list = []
final_prediction  = {}
date_comparison = []
while(date.timestamp()<=end.timestamp()):
    date += datetime.timedelta(days = 1)
    date_comparison.append(date)
    date_list.append(date.timestamp())

from sklearn.preprocessing import PolynomialFeatures
#Initializing and transforming the data:
poly = PolynomialFeatures(degree = 4)
X_Poly = poly.fit_transform(np.array(x_stmps).reshape(len(x_stmps), 1))
y= y.to_numpy()
#poly.fit(X_Poly, y.reshape(len(x), 1))
#Fitting data:
model_linear = LinearRegression()
model_linear.fit(X_Poly, y.reshape(len(x), 1))
#Testing & Visualization:
plt.figure(figsize=(7, 9))

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 

#plotting the predicted data:  
predictions = model_linear.predict(poly.fit_transform(np.array(date_list).reshape(len(date_list), 1)))
plt.plot(  predictions[10:], lw = 3, color = "red", alpha = 0.6)
plt.text(86, 1600, "Infections", fontsize=14, color="red", alpha = 0.6) 
plt.yticks(fontsize = 14)
ax.set_xticks([3, 27 , 58, 88 ])
ax.set_xticklabels(['Sep', 'Oct', 'Nov', 'Dec'])
plt.xticks(fontsize = 14)
plt.title("Predictions For The Number Of infected Cases In Bangladesh (Sep - Dec)", fontsize = 16)

