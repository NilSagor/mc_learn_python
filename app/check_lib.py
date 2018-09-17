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