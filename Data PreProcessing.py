import numpy as npy
import pandas as pd
import matplotlib .pyplot as plot

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #: means columns, the :-1 means not the last one
y = dataset.iloc[:, 3].values #4th column (python like all good languages starts with 0,1,2...)

