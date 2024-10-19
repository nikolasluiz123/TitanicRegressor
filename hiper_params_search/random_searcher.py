import time

from sklearn.model_selection import RandomizedSearchCV

class RandomHipperParamsSearcher:

    def __init__(self,
                 number_iterations: int,
                 n_jobs: int = -1,
                 log_level: int = 0):
        self.number_iterations = number_iterations
        self.n_jobs = n_jobs
        self.log_level = log_level

        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    def search_hipper_parameters(self,
                                 estimator,
                                 params,
                                 data_x,
                                 data_y,
                                 cv,
                                 scoring: str) -> RandomizedSearchCV:
        self.start_search_parameter_time = time.time()

        search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=params,
                                    cv=cv,
                                    n_jobs=self.n_jobs,
                                    verbose=self.log_level,
                                    n_iter=self.number_iterations,
                                    scoring=scoring)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search
