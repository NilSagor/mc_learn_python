
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# load dataset
wine = datasets.load_wine()

'''
Split dataset into training set and test set
70% training and 30% test
'''
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, 
	test_size = 0.3)

'''
Create KNN Classsifier(n_neighbors = 5)
'''
knn = KNeighborsClassifier(n_neighbors = 5)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = knn.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))