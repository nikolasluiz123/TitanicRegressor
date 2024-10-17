from abc import ABC, abstractmethod

from sklearn.model_selection import KFold


class RegressorHipperParamsSearcher(ABC):
    """
    Classe base de pesquisa de hiper parâmetros de algoritimos de classificação do scikit-learn.
    """

    def __init__(self,
                 params: dict[str, list],
                 n_jobs: int = -1,
                 log_level: int = 1):
        """
            :param params: Hiper parâmetros que deseja testar
            :param cv: Estratégia de divisão dos grupos
            :param n_jobs Thread do processador que serão utilizadas
            :param scoring O que o searcher vai utilizar para definir o melhor modelo
            :param log_level Nível dos logs exibidos no processo de busca, varia em 1, 2 e 3
        """

        self.params = params
        self.n_jobs = n_jobs
        self.log_level = log_level

        self.cv = KFold(n_splits=5, shuffle=True)
        self.start_search_parameter_time = 0
        self.end_search_parameter_time = 0

    @abstractmethod
    def search_hipper_parameters(self,
                                 estimator,
                                 data_x,
                                 data_y,
                                 number_iterations: int,
                                 scoring:str='neg_mean_squared_error'):
        """
        Função que deve realizar a busca dos hiper parâmetros.

        :param number_iterations: Número de iterações que o searcher vai realizar. Seu valor é opcional pois há
        implementações que não aceitam um limite de iterações.

        :return: Retorna uma instância de algum BaseSearchCV depois de fazer o fit
        """
        ...
