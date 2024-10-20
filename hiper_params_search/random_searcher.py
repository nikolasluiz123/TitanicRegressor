import time

from sklearn.model_selection import RandomizedSearchCV

class RandomHipperParamsSearcher:
    """
    Implementação para busca de parâmetros utilizando RandomizedSearchCV
    """

    def __init__(self,
                 number_iterations: int,
                 n_jobs: int = -1,
                 log_level: int = 0):
        """
        :param number_iterations: Número de iterações da busca, isso impacta no número de fits realizados com diferentes
        valores.

        :param n_jobs: Número de threads usadas no processamento.

        :param log_level: Nível de log do processo de busca, isso impacta em quanta informação você verá no console.
        """

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
        """
        Função para realizar a busca dos melhores parâmetros do estimador, utilizando RandomizedSearchCV

        :param estimator: Instância do estimador que deseja procurar os parâmetros.

        :param params: Dicionário com os parâmetros e valores que deseja testar.

        :param data_x: Valores de x (features).

        :param data_y: Valores de y (target).

        :param cv: Definição da validação cruzada dentro do processo de busca. Valores que podem ser usados: KFold ou
        StratifiedKFold.

        :param scoring: Métrica avaliada para definição do melhor estimador.

        :return: Retorna a instância de RadomizedSearchCV após os fits.
        """

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
