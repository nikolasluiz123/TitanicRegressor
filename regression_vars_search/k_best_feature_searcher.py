import time

from sklearn.feature_selection import SelectKBest, f_regression

from regression_vars_search.features_searcher import FeaturesSearcher


class SelectKBestFeatureSearcher(FeaturesSearcher):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0,
                 feature_number: int = 3):
        super().__init__(n_jobs, log_level)

        self.feature_numer = feature_number

    def select_features(self, data_x, data_y, cv, scoring: str = 'neg_mean_squared_error', estimator=None):
        self.start_search_features_time = time.time()

        searcher = SelectKBest(score_func=f_regression, k=self.feature_numer)
        searcher.fit_transform(data_x, data_y)

        self.end_search_features_time = time.time()

        return data_x.iloc[:, searcher.get_support()]
