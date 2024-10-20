from abc import ABC, abstractmethod
from typing import Any

from hiper_params_search.random_searcher import RandomHipperParamsSearcher
from manager.history_manager import HistoryManager
from model_validator.validator import ScikitLearnBaseValidator, XGBoostBaseValidator
from regression_vars_search.features_searcher import FeaturesSearcher


class Pipeline(ABC):
    """
    Definição de pipeline de execução utilizado pelos MultiProcessManager.
    """

    def __init__(self,
                 estimator,
                 params,
                 feature_searcher: FeaturesSearcher,
                 params_searcher: RandomHipperParamsSearcher,
                 history_manager: HistoryManager):
        """
        :param estimator: Estimador que deseja validar.

        :param params: Parâmetros e valores que deseja testar.

        :param feature_searcher: Implementação que deseja utilizar para buscar as melhores features.

        :param params_searcher: Implementação que deseja utilizar para buscar o melhor estimador.

        :param history_manager: Implementação que deseja utilizar para manipular o histórico.
        """

        self.estimator = estimator
        self.params = params
        self.feature_searcher = feature_searcher
        self.params_searcher = params_searcher
        self.history_manager = history_manager

    @abstractmethod
    def get_dict_pipeline_data(self):
        """
        Retorna um dicionário contendo os dados do pipeline.
        """


class ScikitLearnPipeline(Pipeline):
    """
    Implementação específica para definir um pipeline para avaliação de modelos do Scikit-Learn.
    """

    def __init__(self,
                 estimator,
                 params,
                 feature_searcher: FeaturesSearcher,
                 params_searcher: RandomHipperParamsSearcher,
                 history_manager: HistoryManager,
                 validator: ScikitLearnBaseValidator):
        super().__init__(estimator, params, feature_searcher, params_searcher, history_manager)

        self.validator = validator

    def get_dict_pipeline_data(self) -> dict[str, Any]:
        return {
            'estimator': type(self.estimator).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }


class XGBoostPipeline(Pipeline):
    """
    Implementação específica para definir um pipeline para avaliação de modelos do XGBoost.
    """

    def __init__(self, estimator,
                 params, feature_searcher: FeaturesSearcher,
                 params_searcher: RandomHipperParamsSearcher,
                 history_manager: HistoryManager,
                 validator: XGBoostBaseValidator):
        super().__init__(estimator, params, feature_searcher, params_searcher, history_manager)

        self.validator = validator

    def get_dict_pipeline_data(self) -> dict[str, Any]:
        return {
            'estimator': type(self.estimator).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }
