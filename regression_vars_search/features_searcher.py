from abc import ABC, abstractmethod


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
        """
        Função utilizada para selecionar as melhores features.

        :param data_x: Valores de x (features).

        :param data_y: Valores de y (target).

        :param cv: Definição da validação cruzada dentro do processo de busca. Valores que podem ser usados: KFold ou
        StratifiedKFold.

        :param scoring: Métrica avaliada para definição do melhor estimador.

        :param estimator: Estimador que será considerado na busca da features.

        :return Retorna um novo data_x contendo apenas as colunas das features selecionadas.
        """