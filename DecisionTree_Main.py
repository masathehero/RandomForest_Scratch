# coding: utf-8
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree_Module import DecisionTree

if __name__ == "__main__":
    # load data
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.20)

    # training and prediction
    dct = DecisionTree(max_depth=None)
    dct.fit(X_train, y_train)
    pred = dct.predict(X_test)

    # evaluation
    evaluation = accuracy_score(pred, y_test)
    print('accuracy: %s' % evaluation)
