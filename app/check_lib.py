#check numpy
import numpy as np 


x = np.array([[1,2,3], [4,5,6]])

print('x:\n {}'.format(x))


#check scipy

from scipy import sparse
#create a 2D numpy array with diagonal of ones and zeros everywhere

eye = np.eye(4)
print('Numpy array:\n {}'.format(eye))





#convert the numpy array to a Scipy sparse matrix in csr format
#only the non-zeroes entries are stored

sparse_matrix = sparse.csr_matrix(eye)
print('\n Scipy SpareNumpy csr_matrix: \n{}'.format(sparse_matrix))


#coo representations

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)

eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO Representations: \n{}'.format(eye_coo))

#%matplotlib inline
import matplotlib.pyplot as plt 

#generate a sequence number from -10 to 10 with 100 steps
x = np.linspace(-10, 10, 100)

#create a second array using sinx
y = np.sin(x)

#the plot functions makes a line chart of one array agnist another
plt.plot(x, y , marker = 'x')
plt.show()

import pandas as pd 
from IPython.display import display

#create a simple dataset of people
data = {'Name': ['John', 'Anna', 'Peter', 'Linda'],
'Location': ['New York', 'Paris', 'Berlin', 'London'],
'Age': [24, 45, 33, 23]
}

data_pandas = pd.DataFrame(data)
#IPython.display allpws pretty printing of dataframes
#in the Jupyter notebook
display(data_pandas)