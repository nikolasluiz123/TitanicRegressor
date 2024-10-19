import time

from sklearn.feature_selection import RFECV

from regression_vars_search.features_searcher import FeaturesSearcher


class RecursiveFeatureSearcher(FeaturesSearcher):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0,
                 min_features: int = 3):
        super().__init__(n_jobs, log_level)

        self.min_feeatures = min_features

    def select_features(self, data_x, data_y, cv, scoring:str='neg_mean_squared_error', estimator=None):
        self.start_search_features_time = time.time()

        searcher = RFECV(estimator=estimator,
                         cv=cv,
                         scoring=scoring,
                         n_jobs=self.n_jobs,
                         verbose=self.log_level,
                         min_features_to_select=self.min_feeatures)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.support_]
