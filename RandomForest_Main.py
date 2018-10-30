# coding: utf-8
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest_Module import RandomForest

if __name__ == '__main__':
    # load data
    rs = 2  # None
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.20, random_state=rs)

    # training and prediction
    rf = RandomForest(n_estimators=10, max_depth=None, tree='scratch', random_state=rs)
    # rf = RandomForest(n_estimators=10, max_depth=None, tree='sklearn', random_state=rs)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    # evaluation
    evaluation = accuracy_score(pred, y_test)
    print('accuracy: %s' % evaluation)
