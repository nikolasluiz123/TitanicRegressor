import numpy as np
import pandas as pd
from tabulate import tabulate

from hiper_params_search.searcher import RegressorHipperParamsSearcher
from manager.history_manager import HistoryManager
from model_validator.result import ValidationResult
from model_validator.validator import BaseValidator
from regression_vars_search.features_searcher import FeaturesSearcher


class ProcessManager:
    """
    Classe responsável por centralizar os processos necessários para obter um modelo de machine learning utilizando
    o scikit-learn.
    """

    def __init__(self,
                 data_x,
                 data_y,
                 estimator,
                 seed: int,
                 feature_searcher: FeaturesSearcher,
                 params_searcher: RegressorHipperParamsSearcher,
                 validator: BaseValidator,
                 history_manager: HistoryManager,
                 scoring: str = 'neg_mean_squared_error',
                 save_history: bool = True,
                 history_index: int = None):
        self.data_x = data_x
        self.data_y = data_y
        self.estimator = estimator
        self.feature_searcher = feature_searcher
        self.params_searcher = params_searcher
        self.validator = validator
        self.history_manager = history_manager
        self.scoring = scoring
        self.save_history = save_history
        self.history_index = history_index

        np.random.seed(seed)

    def process(self, number_interations: int):
        """
        Função que realiza todos os processos para obter um modelo

        :param number_interations: Número de iterações, utilizado apenas quando a implementação do BaseSearchCV aceita
        """
        self.__process_feature_selection()

        search_cv = self.__process_hiper_params_search(number_interations)
        validation_result = self.__process_validation(search_cv)

        self.__save_data_in_history(validation_result)
        self.__show_results(validation_result)

    def __process_feature_selection(self):
        self.data_x = self.feature_searcher.select_features(
            estimator=self.estimator,
            data_x=self.data_x,
            data_y=self.data_y,
            scoring=self.scoring
        )

        print()
        print(f'Features selecionadas')
        print(tabulate(self.data_x.head(), headers='keys', tablefmt='psql', showindex=False))
        print()

    def __process_hiper_params_search(self, number_interations: int):
        """
        Função para realizar a busca dos melhores parâmetros. Se for informado valor em history_index esse processo
        não precisa ser executado pois será pego do histórico.

        :param number_interations: Número de iterações, utilizado apenas quando a implementação do BaseSearchCV aceita
        """
        if self.history_index is None:
            return self.params_searcher.search_hipper_parameters(
                estimator=self.estimator,
                data_x=self.data_x,
                data_y=self.data_y,
                number_iterations=number_interations,
                scoring=self.scoring
            )
        else:
            return None

    def __process_validation(self, search_cv) -> ValidationResult:
        """
        Função que realiza a validação do modelo obtido. Se a instância do BaseSearchCV não for passada significa que
        não há necessidade de executar a validação e será retornado o objeto obtido do histórico.

        :param search_cv: Implementação de BaseSearchCV
        """
        if search_cv is None:
            return self.history_manager.load_result_from_history(self.history_index)
        else:
            return self.validator.validate(searcher=search_cv,
                                           data_x=self.data_x,
                                           data_y=self.data_y,
                                           scoring=self.scoring)

    def __save_data_in_history(self, result: ValidationResult):
        """
        Função para salvar os dados no histórico se save_history for True e a execução não tiver sido com base em um
        registro que já esta no histórico.

        :param result: Objeto com as métricas matemáticas calculadas
        """
        if self.save_history and self.history_index is None:
            feature_selection_time = self.feature_searcher.end_search_features_time - self.feature_searcher.start_search_features_time
            search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
            validation_time = self.validator.end_best_model_validation - self.validator.start_best_model_validation

            self.history_manager.save_result(result,
                                             feature_selection_time=self.__format_time(feature_selection_time),
                                             search_time=self.__format_time(search_time),
                                             validation_time=self.__format_time(validation_time),
                                             scoring=self.scoring,
                                             features=self.data_x.columns.tolist())

    def __show_results(self, result: ValidationResult):
        """
        Função para exibir todas as métricas no console

        :param result: Objeto com as métricas matemáticas calculadas
        """

        result.show_cross_val_metrics()

        feature_selection_time = self.feature_searcher.end_search_features_time - self.feature_searcher.start_search_features_time
        search_time = self.params_searcher.end_search_parameter_time - self.params_searcher.start_search_parameter_time
        validation_time = self.validator.end_best_model_validation - self.validator.start_best_model_validation

        print(f'Tempo da Busca de Features              : {self.__format_time(feature_selection_time)}')
        print(f'Tempo da Busca de Parâmetros            : {self.__format_time(search_time)}')
        print(f'Tempo da Validação                      : {self.__format_time(validation_time)}')

    @staticmethod
    def __format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
