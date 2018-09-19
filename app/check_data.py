from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

import pandas as pd
import numpy as np 
# import mglearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt 


iris_dataset = load_iris()

 

X_train, X_test, y_train, y_test = train_test_split(
	iris_dataset['data'], 
	iris_dataset['target'], 
	random_state = 0)


#create dataframe from data in X_train
#lebel the columns using  the strings in iris_dataset.feature_names

iris_dataframe = pd.DataFrame(X_train, columns = iris_dataset.feature_names)

#create scatter matrix from dataframe, color by y_train
grr = pd.plotting.scatter_matrix(
		iris_dataframe, 
		c = y_train, 
		figsize = (15, 15), marker = 'o',
		hist_kwds = {'bins': 20}, 
		s = 60, 
		alpha = .8)
plt.show()