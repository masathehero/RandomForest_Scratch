import numpy as np
from DecisionTree_Module import DecisionTree


class RandomForest():
    def __init__(self, n_estimators=5, max_depth=None, tree='scratch',
                 random_state=None, ind_size=None, col_size=None):
        assert tree in ['scratch', 'sklearn'], 'tree argument >> ["scratch", "sklearn"]'
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tree = tree
        self.random_state = random_state
        self.ind_size = ind_size
        self.col_size = col_size

    def _boot_strap(self, X, y):
        '''ブートストラップサンプリング'''
        np.random.seed(self.random_state)
        record_size, feat_size = X.shape
        # サンプルサイズが指定されてなければ、元データのルートを取る
        if self.ind_size is None:
            self.ind_size = int(np.ceil(np.sqrt(record_size)))
        if self.col_size is None:
            self.col_size = int(np.ceil(np.sqrt(feat_size)))
        # サンプリングするindexとcolumnを決める
        sample_ind = np.random.choice(record_size, self.ind_size, replace=False)
        sample_col = np.random.choice(feat_size, self.col_size, replace=False)
        # サンプリングの実行
        sample_X = X[sample_ind]
        sample_X = sample_X[:, sample_col]
        sample_y = y[sample_ind]
        return sample_X, sample_y, sample_col

    def fit(self, X, y):
        '''学習'''
        self.clf_ls = list()
        self.sample_col_ls = list()
        for i in range(self.n_estimators):
            sample_X, sample_y, sample_col = self._boot_strap(X, y)
            if self.tree == 'scratch':
                clf = DecisionTree(max_depth=self.max_depth)
            elif self.tree == 'sklearn':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(random_state=self.random_state)
            clf.fit(sample_X, sample_y)
            self.clf_ls.append(clf)  # 学習器の格納
            self.sample_col_ls.append(sample_col)  # サンプリングした列の格納

    def predict(self, X):
        '''予測'''
        # まずは、各学習器で予測
        predict_ls = list()
        for clf, sample_col in zip(self.clf_ls, self.sample_col_ls):
            sample_testX = X[:, sample_col]
            tree_pred = clf.predict(sample_testX)
            predict_ls.append(tree_pred)
        predictions = np.array(predict_ls)
        # 各学習器の予測で、多数決を取る
        final_pred_ls = list()
        for i in range(len(predictions[0])):
            final_pred = np.argmax(np.bincount(predictions[:, i]))
            final_pred_ls.append(final_pred)
        return final_pred_ls
