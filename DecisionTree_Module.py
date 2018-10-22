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
    def _split_ans(self, feat, ans, thresh):
        '''教師データを、特徴量のある特定のthreshで分ける'''
        feat = np.array(feat)
        ans = np.array(ans)
        ind = feat < thresh
        small_leaf_ans = ans[ind]
        large_leaf_ans = ans[~ind]
        return small_leaf_ans, large_leaf_ans

    def _count_leaf_size(self, leaf_ans):
        '''そのノードで、各クラスのデータが何個存在するかカウント'''
        leaf_ans = np.array(leaf_ans)
        unique_class = list(set(leaf_ans))
        cl_size = [np.sum(leaf_ans == cls) for cls in unique_class]
        return cl_size

    # 上の二つのモジュールを使う
    def _cal_feature_gini(self, feat, ans, thresh):
        '''ある特徴量で、あるthreshの時のジニ係数の計算'''
        small_leaf, large_leaf = self._split_ans(feat, ans, thresh)
        cl_dist1 = self._count_leaf_size(large_leaf)
        cl_dist2 = self._count_leaf_size(small_leaf)
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
                    thresh = thresh_list[single_feat_gini.argmin()]
                    best_split_point = (feat_col, thresh, smallest_gini)
        # print('splitting point: feature={0}, thresh={1}, gini={2}'.format(
        #    best_split_point[0], best_split_point[1], best_split_point[2]))
        return best_split_point

    def split_data(self, features, ans):
        '''特徴量、及び教師ラベルを、best_pointでsplit'''
        if len(set(ans)) > 1:
            feat_col, thresh, _ = self._best_split_point(features, ans)
            ind = features[:, feat_col] < thresh
            small_features = list([features[ind], ans[ind]])
            large_features = list([features[~ind], ans[~ind]])
            return small_features, large_features, feat_col, thresh
        else:
            org_features = list([features, ans])
            return org_features


class DecisionTree(DataSplitter):
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self._depth = 0
        self._pred_depth = 0

    def fit(self, X, y):
        '''決定木の訓練'''
        if self._depth == 0:
            self._params_ls = list()
            X, y = list([X]), list([y])
        # print('Depth: %s,' % self._depth, end='\n')

        X_new, y_new = list(), list()
        for feat, ans in zip(X, y):
            if len(set(ans)) == 1:
                X_new.append(feat)
                y_new.append(ans)
                feat_col, thresh = None, None
                cls_no = np.argmax(np.bincount(ans))
                self._params_ls.append((self._depth, feat_col, thresh, cls_no))
            else:
                small_features, large_features, feat_col, thresh = self.split_data(feat, ans)
                X_new.append(small_features[0])
                y_new.append(small_features[1])
                X_new.append(large_features[0])
                y_new.append(large_features[1])
                cls_no = None
                self._params_ls.append((self._depth, feat_col, thresh, cls_no))
        self._depth += 1
        # 終了条件: 全てのノードが、単一クラスになった時
        if (len(X) == len(X_new)) | (self._depth == self.max_depth):
            self.params = np.array(self._params_ls)
            '''
            if (len(X) == len(X_new)):
                print('all leaves converged')
            else:
                print('reached max depth')
            '''
        else:
            return self.fit(X_new, y_new)

    def predict(self, X):
        if self._pred_depth == 0:
            # 変数の初期化
            self.result_ind = np.array([])
            self.result_cls_no = np.array([])
            ind_no = np.arange(len(X)).reshape(len(X), 1)
            X = list([np.concatenate([X, ind_no], axis=1)])
        # この階層で使うパラメータをだけを引っ張ってくる
        valid_params = self.params[self.params[:, 0] == self._pred_depth]
        Xpred = list()
        for feat, vp in zip(X, valid_params):
            _, feat_col, thresh, cls_no = vp
            if cls_no is not None:
                # cls_noがNoneではないleafは収束している
                if feat.shape[0] >= 1:
                    cls_no_full = np.full(len(feat), cls_no)
                    self.result_ind = np.append(self.result_ind, feat[:, -1])
                    self.result_cls_no = np.append(self.result_cls_no, cls_no_full)
                # 次のzip(X, valid_params)でsizeが合うようにpadding
                Xpred.append(feat)
                continue
            ind = feat[:, feat_col] < thresh
            # 次の再帰に渡すXを保存
            feat_small, feat_large = feat[ind], feat[~ind]
            Xpred.append(feat_small)
            Xpred.append(feat_large)
        self._pred_depth += 1
        # 終了条件：fitした時の_depthに到達したら
        if self._pred_depth >= self._depth:
            self._pred_depth = 0
            result = np.conj([self.result_ind, self.result_cls_no])
            result = np.unique(result, axis=1)[1].astype(int)
            return result
        else:
            return self.predict(Xpred)
