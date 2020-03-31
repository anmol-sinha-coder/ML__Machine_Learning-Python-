# Simple Linear Regression
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
          # Importing the libraries

# Importing the dataset
med = pd.read_csv('counterfeit_train.csv')

mean_value=med['Counterfeit_Weight'].mean()
med['Counterfeit_Weight']=med['Counterfeit_Weight'].fillna(mean_value)
print(med.isnull().sum())

med.drop(['Medicine_ID'],1,inplace=True)
med.drop(['DistArea_ID'],1,inplace=True)
#med.convert_objects(convert_numeric=True)

med_onehot = med.copy()
med_onehot = pd.get_dummies(med_onehot, columns=['SidEffect_Level'], prefix = ['SideEffect'])

med_onehot = med_onehot.copy()
med_onehot = pd.get_dummies(med_onehot, columns=['Medicine_Type'], prefix = ['MedType'])

med_onehot = med_onehot.copy()
med_onehot = pd.get_dummies(med_onehot, columns=['Area_Type'], prefix = ['ArType'])

med_onehot = med_onehot.copy()
med_onehot = pd.get_dummies(med_onehot, columns=['Area_City_Type'], prefix = ['Tier'])

med_onehot = med_onehot.copy()
med_onehot = pd.get_dummies(med_onehot, columns=['Area_dist_level'], prefix = ['DistLevel'])
med1 = med_onehot

y = med1['Counterfeit_Sales']
X = med1.drop("Counterfeit_Sales", axis=1)

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_train)
print(y_pred)
