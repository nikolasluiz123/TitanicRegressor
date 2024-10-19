from abc import ABC, abstractmethod

from sklearn.model_selection import KFold


class FeaturesSearcher(ABC):

    def __init__(self,
                 n_jobs: int = -1,
                 log_level: int = 0):
        self.n_jobs = n_jobs
        self.log_level = log_level

        self.start_search_features_time = 0
        self.end_search_features_time = 0

    @abstractmethod
    def select_features(self, data_x, data_y, cv, scoring:str='neg_mean_squared_error', estimator=None):
        ...