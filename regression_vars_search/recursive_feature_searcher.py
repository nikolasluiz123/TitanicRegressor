import time

from sklearn.feature_selection import RFECV

from regression_vars_search.features_searcher import FeaturesSearcher


class RecursiveFeatureSearcher(FeaturesSearcher):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 1):
        super().__init__(n_jobs, log_level)

    def select_features(self, estimator, data_x, data_y, scoring:str='neg_mean_squared_error'):
        self.start_search_features_time = time.time()

        searcher = RFECV(estimator=estimator,
                         cv=self.cv,
                         scoring=scoring,
                         n_jobs=self.n_jobs,
                         verbose=self.log_level)
        searcher = searcher.fit(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.support_]
