import numpy as np


class GiniCalculater():
    def _cal_gini(self, cl_dist):
        '''ジニ係数の計算'''
        leaf_size = np.array(cl_dist).sum()
        ratio_list = [(cl_size/leaf_size)**2 for cl_size in cl_dist]
        single_gini = 1 - np.array(ratio_list).sum()
        return single_gini, leaf_size

    def average_gini(self, cl_dist1, cl_dist2):
        '''二つのデータのジニ係数の加重平均'''
        gini_val1, leaf_size1 = self._cal_gini(cl_dist1)
        gini_val2, leaf_size2 = self._cal_gini(cl_dist2)
        gini_ls = [gini_val1, gini_val2]
        leaf_size_ls = [leaf_size1, leaf_size2]
        ave_gini = np.average(gini_ls, weights=leaf_size_ls)
        return ave_gini


class DataSplitter(GiniCalculater):
    def _count_leaf_size(self, leaf_ans):
        '''そのノードで、各クラスのデータが何個存在するかカウント'''
        leaf_ans = np.array(leaf_ans)
        unique_class = list(set(leaf_ans))
        cl_dist = [np.sum(leaf_ans == cls) for cls in unique_class]
        return cl_dist

    def _cal_feature_gini(self, feat, ans, thresh):
        '''ある特徴量でのあるthreshで分割した時のジニ係数の計算'''
        # まず、threshでleafを分ける
        ind = feat < thresh
        small_leaf_ans = ans[ind]
        large_leaf_ans = ans[~ind]
        # ジニ係数を計算
        cl_dist1 = self._count_leaf_size(large_leaf_ans)
        cl_dist2 = self._count_leaf_size(small_leaf_ans)
        gini = self.average_gini(cl_dist1, cl_dist2)
        return gini

    def _best_split_point(self, features, ans):
        '''分けるべき、特徴量(feat_col)と臨界値(thresh)を決定'''
        smallest_gini = 1
        for feat_col in range(len(features[0])):
            feat = features[:, feat_col]
            thresh_list = np.unique(feat)
            single_feat_gini = np.array(
                    [self._cal_feature_gini(feat, ans, thresh) for thresh in thresh_list])
            if smallest_gini > single_feat_gini.min():
                    smallest_gini = single_feat_gini.min()
                    # issue: threshの決め方は工夫の余地あり
                    thresh = thresh_list[single_feat_gini.argmin()]
                    best_split_point = (feat_col, thresh, smallest_gini)
        return best_split_point

    def split_data(self, features, ans, cand_features):
        '''特徴量、及び教師ラベルを、best_pointでsplit'''
        # issue: feature_importance計算するならsmallest_giniも必要だけど、今回は割愛
        np.random.seed(self.random_state)
        if cand_features is not None:
            candidate_col = list(np.random.choice(len(features[0]), cand_features, replace=False))
        else:
            candidate_col = list(np.arange(len(features[0])))
        cand_features = features[:, candidate_col]
        feat_col, thresh, _ = self._best_split_point(cand_features, ans)
        ind = cand_features[:, feat_col] < thresh
        small_features = list([features[ind], ans[ind]])
        large_features = list([features[~ind], ans[~ind]])
        split_point = (candidate_col, feat_col, thresh)
        return small_features, large_features, split_point


class DecisionTree(DataSplitter):
    def __init__(self, max_depth=None, cand_features=None, random_state=None):
        self.max_depth = max_depth
        self.cand_features = cand_features
        self.random_state = random_state
        self._depth = 0
        self._pred_depth = 0

    def fit(self, X, y):
        '''決定木の訓練'''
        if self._depth == 0:
            self._params_ls = list()
            X, y = list([X]), list([y])
        # print('Depth: %s,' % self._depth, end='\n')

        X_new, y_new = list(), list()
        is_max_depth = (self._depth == self.max_depth)
        for feat, ans in zip(X, y):
            small_features, large_features, split_point = self.split_data(feat, ans, self.cand_features)
            candidate_col, feat_col, thresh = split_point
            is_leaf_converge = (len(set(ans)) == 1)
            is_no_unique_feature = (len(np.unique(feat[:, candidate_col], axis=0)) == 1)
            if is_max_depth | is_leaf_converge | is_no_unique_feature:
                # 収束したleafについて、パラメータの保存
                cls_no = np.argmax(np.bincount(ans))  # leafのクラス
            else:
                # 収束してないleafについては、再びsplitする
                X_new = X_new + [small_features[0]] + [large_features[0]]
                y_new = y_new + [small_features[1]] + [large_features[1]]
                cls_no = None
            self._params_ls.append((self._depth, candidate_col, feat_col, thresh, cls_no))
        # 終了条件: 一回もleafがsplitされなかった時
        if len(X_new) == 0:
            self.params = np.array(self._params_ls)
        else:
            self._depth += 1
            return self.fit(X_new, y_new)

    def predict(self, X):
        if self._pred_depth == 0:
            # 変数の初期化
            self.result_ind = np.array([])
            self.result_cls_no = np.array([])
            self.Xind_col = len(X[0])
            ind_no = np.arange(len(X)).reshape(len(X), 1)
            X = list([np.concatenate([X, ind_no], axis=1)])  # 元のindex_noを付けておく
        # この階層で使うパラメータだけを引っ張ってくる
        valid_params = self.params[self.params[:, 0] == self._pred_depth]
        Xpred = list()
        for feat, vp in zip(X, valid_params):
            _, candidate_col, feat_col, thresh, cls_no = vp
            if cls_no is not None:
                # 収束したleafの結果を保存 (cls_noがNoneではないleafは収束している)
                if feat.shape[0] >= 1:
                    cls_no_full = np.full(len(feat), cls_no)
                    self.result_ind = np.append(self.result_ind, feat[:, -1])
                    self.result_cls_no = np.append(self.result_cls_no, cls_no_full)
            else:
                # 次の再帰に渡すXを保存
                candidate_col.append(self.Xind_col)
                cand_feat = feat[:, candidate_col]
                ind = cand_feat[:, feat_col] < thresh
                feat_small, feat_large = feat[ind], feat[~ind]
                Xpred = Xpred + [feat_small] + [feat_large]
        # 終了条件：fitした時のdepthに到達したら
        if self._pred_depth >= self._depth:
            self._pred_depth = 0
            result = np.conj([self.result_ind, self.result_cls_no])
            result = np.unique(result, axis=1)[1].astype(int)  # np.uniqueでソートもしてくれる。
            return result
        else:
            self._pred_depth += 1
            return self.predict(Xpred)
