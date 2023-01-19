#Support vector machine

import sklearn
from sklearn import svm
from sklearn import datasets
cancer = datasets.load_breast_cancer()
print("Features: ", cancer.feature_names)
print("Labels: ", cancer.target_names)

x = cancer.data  # All of the features
y = cancer.target  # All of the labels

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

print(x_train[:5], y_train[:5])

from sklearn import svm

clf = svm.SVC()
clf.fit(x_train, y_train)

from sklearn import metrics

y_pred = clf.predict(x_test) # Predict values for our test data

acc = metrics.accuracy_score(y_test, y_pred) # Test them against our correct values
print(acc)


clf = svm.SVC(kernel="linear")

clf = svm.SVC(kernel="linear", C=2)


