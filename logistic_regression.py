# Author: Priti Gupta
# Date: June 16th, 2023
# Description: Using logistic regression to detect breast cancer, aiming to develop a predictive model.
# GitHub: https://github.com/PritiG1/logistic-regression-breast-cancer-classification

import pandas as pd
import pickle

"""## Importing the dataset"""

dataset = pd.read_csv('breast_cancer.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""## Training the Logistic Regression model on the Training set"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

"""## Computing the accuracy with k-Fold Cross Validation"""

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


## predicting class
print('The tumor belongs to class ',classifier.predict([[5, 1, 1, 1, 2, 1, 3, 1, 1]])) 

# Save the model to a file
with open('classification_model_cancer.pkl', 'wb') as file:
    pickle.dump(classifier, file)