import numpy as np


class RandomForest():
    def __init__(self, n_estimators=5, max_depth=None, tree='scratch',
                 random_state=None, ind_size=None, cand_features=None):
        assert tree in ['scratch', 'sklearn'], 'tree argument >> ["scratch", "sklearn"]'
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.tree = tree
        self.random_state = random_state
        self.ind_size = ind_size
        self.cand_features = cand_features

    def _boot_strap(self, X, y):
        '''ブートストラップサンプリング'''
        np.random.seed(self.random_state)
        # サンプリングするindexを決める。
        sample_ind = np.random.choice(self._record_size, self.ind_size, replace=False)
        sample_X = X[sample_ind]
        sample_y = y[sample_ind]
        return sample_X, sample_y

    def fit(self, X, y):
        '''学習'''
        # サンプルサイズが指定されてなければ、元データのルートを取る
        self._record_size, self._feat_size = X.shape
        if self.ind_size is None:
            self.ind_size = int(np.ceil(np.sqrt(self._record_size)))
        if self.cand_features is None:
            self.cand_features = int(np.ceil(np.sqrt(self._feat_size)))

        self.clf_ls = list()
        for i in range(self.n_estimators):
            sample_X, sample_y = self._boot_strap(X, y)
            if self.tree == 'scratch':
                from DecisionTree_Module import DecisionTree
                clf = DecisionTree(
                        max_depth=self.max_depth,
                        cand_features=self.cand_features,
                        random_state=self.random_state)
            elif self.tree == 'sklearn':
                from sklearn.tree import DecisionTreeClassifier
                clf = DecisionTreeClassifier(
                        max_depth=self.max_depth,
                        max_features=self.cand_features,
                        random_state=self.random_state)
            clf.fit(sample_X, sample_y)
            self.clf_ls.append(clf)  # 学習器の格納

    def predict(self, X):
        '''予測'''
        # まずは、各学習器で予測
        predict_ls = list()
        for clf in self.clf_ls:
            tree_pred = clf.predict(X)
            predict_ls.append(tree_pred)
        predictions = np.array(predict_ls)
        # 各学習器の予測で、多数決を取る
        final_pred_ls = list()
        for i in range(len(predictions[0])):
            final_pred = np.argmax(np.bincount(predictions[:, i]))
            final_pred_ls.append(final_pred)
        return final_pred_ls
