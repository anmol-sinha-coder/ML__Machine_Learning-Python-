import pandas as pd
import numpy as np

social = pd.read_csv("traincodepth.csv")
social.describe()
X = social.iloc[:,:-1].values
'''
a = social.iloc[: , 8].values
b = social.iloc[: , 12].values
'''
c = social.iloc[: , 10].values
d = social.iloc[: , 16].values

from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN',strategy = 'mean') #NaN replaced by mean values
imputer = imputer.fit(X[:,[1,2,3,4,5,6,7,9,11,13,14,15,17,18,19,20]])
X[:,[1,2,3,4,5,6,7,9,11,13,14,15,17,18,19,20]] =imputer.transform(X[:,[1,2,3,4,5,6,7,9,11,13,14,15,17,18,19,20]])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,8] = labelencoder_X.fit_transform(X[:,8])#Only labels for 'Country' column
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,12] = labelencoder_X.fit_transform(X[:,12])
X[:,16] = labelencoder_X.fit_transform(X[:,16])#Only labels for 'Country' column

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray() #Onehotencoder cannot accept strings, so labelencoder used

labelencoder_c = LabelEncoder() #Redundancy is eliminated, no need of onehotencoder
c = labelencoder_c.fit_transform(c)
labelencoder_d = LabelEncoder() #Redundancy is eliminated, no need of onehotencoder
d = labelencoder_d.fit_transform(d)


print(X)