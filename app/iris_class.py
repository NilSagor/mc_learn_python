'''
A first application: Classifying iris species

'''

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print("Keys of iris_dataset: {}".format(iris_dataset.keys()))