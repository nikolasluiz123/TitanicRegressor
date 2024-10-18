from abc import ABC, abstractmethod

from sklearn.model_selection import KFold


class FeaturesSearcher(ABC):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 1):
        self.n_jobs = n_jobs
        self.log_level = log_level

        self.cv = KFold(n_splits=5, shuffle=True)
        self.start_search_features_time = 0
        self.end_search_features_time = 0

    @abstractmethod
    def select_features(self, data_x, data_y, estimator=None, scoring:str='neg_mean_squared_error'):
        ...