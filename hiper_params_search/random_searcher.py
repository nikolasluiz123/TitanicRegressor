import time

from sklearn.model_selection import RandomizedSearchCV

from hiper_params_search.searcher import RegressorHipperParamsSearcher


class RegressorRandomHipperParamsSearcher(RegressorHipperParamsSearcher):
    """
    Classe específica para buscar hiper parâmetros utilizando o metodo de pesquisa aleatória, onde um número específico
    de combinações será testada
    """

    def __init__(self,
                 number_iterations: int,
                 params: dict[str, list],
                 n_jobs: int = -1,
                 log_level: int = 1):
        super().__init__(number_iterations, params, n_jobs, log_level)

    def search_hipper_parameters(self,
                                 estimator,
                                 data_x,
                                 data_y,
                                 cv,
                                 scoring:str='neg_mean_squared_error') -> RandomizedSearchCV:
        self.start_search_parameter_time = time.time()

        search = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=self.params,
                                    cv=cv,
                                    n_jobs=self.n_jobs,
                                    verbose=self.log_level,
                                    n_iter=self.number_iterations,
                                    scoring=scoring)

        search.fit(X=data_x, y=data_y)

        self.end_search_parameter_time = time.time()

        return search
