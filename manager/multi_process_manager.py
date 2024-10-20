from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
import pandas as pd
import xgboost as xgb
from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from tabulate import tabulate

from manager.history_manager import HistoryManager
from manager.multi_process_manager_pipelines import ScikitLearnPipeline, XGBoostPipeline, Pipeline
from model_validator.result import ValidationResult

T = TypeVar('T', bound=Pipeline)


class MultiProcessManager(ABC):
    """
    Classe responsável por controlar a execução dos processos necessários para validação de N estimadores, um após o outro.

    Com essa implementação é possível avaliar estimadores diferentes de uma biblioteca e ao fim do processo escolher o
    melhor entre eles para utilizar em alguma feature desejada.
    """

    def __init__(self,
                 data_x,
                 data_y,
                 seed: int,
                 fold_splits: int,
                 pipelines: list[T] | T,
                 history_manager: HistoryManager,
                 stratified: bool = False,
                 scoring: str = 'neg_mean_squared_error',
                 save_history: bool = True,
                 history_index: int = None):
        """
        :param data_x: Valores de x (features).

        :param data_y: Valores de y (target).

        :param seed: Seed utilizada no random do np. Isso possibilita reprodutibilidade.

        :param fold_splits: Quantidade de dados por fold (grupo) para realização da validação cruzada.

        :param pipelines: Pode ser uma lista ou um único Pipeline, contendo as implementações específicas dos processos
        que deseja realizar para validar o modelo.

        :param history_manager: Implementação do HistoryManager para lidar com o melhor estimador encontrado após testar
        todos da lista de Pipelines.

        :param stratified: Flag para indicar se deve ou não utilizar StratifiedKFold.

        :param scoring: Métrica avaliada para definição do melhor estimador.

        :param save_history: Flag que indica se o resultado do processo deve ser salvo no historico ou não.

        :param history_index: Índice da lista de histórico que será usado para recuperar algum resultado específico e
        reutilizá-lo.
        """

        self.data_x = data_x
        self.data_y = data_y
        self.pipelines = pipelines
        self.history_manager = history_manager
        self.scoring = scoring
        self.save_history = save_history
        self.history_index = history_index

        self.results = []

        np.random.seed(seed)

        if stratified:
            self.cv = StratifiedKFold(n_splits=fold_splits, shuffle=True)
        else:
            self.cv = KFold(n_splits=fold_splits, shuffle=True)

    @abstractmethod
    def _process_validation(self, pipeline: T, search_cv: RandomizedSearchCV) -> ValidationResult:
        """
        Função que realiza o processo de validação do estimador.

        :param pipeline: Pipeline definido que contem o Validator que será utilizado pela função.

        :param search_cv: Instância treinada de RandomizedSearchCV que já encontrou os melhores parâmetros.

        :return: Retorna uma implementação de ValidationResult com os resultados da validação.
        """

    def _on_after_process_pipelines(self, df_results: DataFrame):
        """
        Função opcional que pode ser implementada quando desejar exeutar algum processo após todos os pipelines terem
        sido executados e os resultados individuais obtidos.

        :param df_results: DataFrame com os resultados dos melhores estimadores encontrados.
        """

    def process_pipelines(self):
        """
        Função utilizada para iniciar o processamento dos pipelines definidos.
        """

        if type(self.pipelines) is list:
            for pipeline in self.pipelines:
                self._process_single_pipeline(pipeline)
        else:
            self._process_single_pipeline(self.pipelines)

        df_results = self._show_results()
        self._on_after_process_pipelines(df_results)

    def _process_single_pipeline(self, pipeline):
        """
        Função que executa os processos necessários que estão presentes dentro de um Pipeline

        :param pipeline: Pipeline que será executado.
        """

        self._process_feature_selection(pipeline)

        search_cv = self._process_hiper_params_search(pipeline)
        validation_result = self._process_validation(pipeline, search_cv)

        self._save_data_in_history(pipeline, validation_result)
        self._append_new_result(pipeline, validation_result)

    def _process_feature_selection(self, pipeline: T):
        """
        Função para selecionar as melhores features que serão colocadas no data_x utilizando a implementação definida
        no pipeline. Isso vai eliminar dados que serão considerados como irrelevantes para o estimador.

        Esse processo só deve ser executado quando não desejar reutilizar dados históricos, ou seja, não pode ser passado
        um valor para history_index. Se estamos reutilizando valores do histórico não faz diferença sobrescrever data_x
        pois nenhum processamento será executado, apenas carregamos os dados do JSON.

        :param pipeline: Pipeline que será executado.
        """

        if self.history_index is None:
            self.data_x = pipeline.feature_searcher.select_features(
                estimator=pipeline.estimator,
                data_x=self.data_x,
                data_y=self.data_y,
                scoring=self.scoring,
                cv=self.cv
            )

    def _process_hiper_params_search(self, pipeline: T) -> RandomizedSearchCV | None:
        """
        Função para buscar os melhores parâmetros do estimador utilizando RandomizedSearchCV definido no pipeline.

        Se for definido um history_index não será feita a busca pois os dados serão recuperados do JSON.

        :param pipeline: Pipeline que será executado.

        :return: Retorna uma instância de RandomizedSearchCV treinada que já contém o melhor estimador. Pode ser None
        se for definido o history_index.
        """

        if self.history_index is None:
            return pipeline.params_searcher.search_hipper_parameters(
                estimator=pipeline.estimator,
                params=pipeline.params,
                data_x=self.data_x,
                data_y=self.data_y,
                scoring=self.scoring,
                cv=self.cv
            )
        else:
            return None

    def _save_data_in_history(self, pipeline: T, result: ValidationResult):
        """
        Função para salvar os dados no histórico, utilizando o manager definido no pipeline. Só vai salvar no histórico
        se não for fornecido history_index, isso vai evitar salvar dados repetidos.

        :param pipeline: Pipeline que será executado.

        :param result: Resultado da função _process_validation.
        """

        if self.save_history and self.history_index is None:
            feature_selection_time, search_time, validation_time = self._get_execution_times(pipeline)

            pipeline.history_manager.save_result(result,
                                                 feature_selection_time=self._format_time(feature_selection_time),
                                                 search_time=self._format_time(search_time),
                                                 validation_time=self._format_time(validation_time),
                                                 scoring=self.scoring,
                                                 features=self.data_x.columns.tolist())

    def _get_execution_times(self, pipeline):
        feature_selection_time = pipeline.feature_searcher.end_search_features_time - pipeline.feature_searcher.start_search_features_time
        search_time = pipeline.params_searcher.end_search_parameter_time - pipeline.params_searcher.start_search_parameter_time
        validation_time = pipeline.validator.end_best_model_validation - pipeline.validator.start_best_model_validation
        return feature_selection_time, search_time, validation_time

    def _append_new_result(self, pipeline: T, result: ValidationResult):
        """
        Função para adicionar um resultado da validação a uma lista interna que, ao fim de todos os processos, é exibida
        em um DataFrame para que possam ser visualizados os resultados de todos os estimadores avaliados.

        Se for passado um history_index ao invés de refazer os cálculo carregamos isso do histórico.

        :param pipeline: Pipeline que será executado.

        :param result: Resultado da função _process_validation.
        """

        pipeline_infos = pipeline.get_dict_pipeline_data()
        performance_metrics = result.append_data(pipeline_infos)

        if self.history_index is None:
            self._calculate_processes_time(performance_metrics, pipeline)
        else:
            self._load_processes_time_from_history(performance_metrics, pipeline)

        self.results.append(performance_metrics)

    def _calculate_processes_time(self, performance_metrics, pipeline: T):
        """
        Função para adicionar nas métricas de performance (um dicionário) os tempos de processamento do pipeline.

        :param performance_metrics: Dicionário com algumas métricas já adicionadas

        :param pipeline: Pipeline que será executado.
        """
        feature_selection_time, search_time, validation_time = self._get_execution_times(pipeline)

        performance_metrics['feature_selection_time'] = self._format_time(feature_selection_time)
        performance_metrics['search_time'] = self._format_time(search_time)
        performance_metrics['validation_time'] = self._format_time(validation_time)

    def _load_processes_time_from_history(self, performance_metrics, pipeline: T):
        """
        Função para realizar o carregamento de informações de performance do histórico, as quais não estão presentes
        na implementação de ValidatorResult.

        :param performance_metrics: Dicionário com algumas métricas já adicionadas

        :param pipeline: Pipeline que será executado.
        """
        history_dict = pipeline.history_manager.get_dictionary_from_json(self.history_index)

        performance_metrics['feature_selection_time'] = history_dict['feature_selection_time']
        performance_metrics['search_time'] = history_dict['search_time']
        performance_metrics['validation_time'] = history_dict['validation_time']

    def _show_results(self) -> DataFrame:
        """
        Exibe o DataFrame com os resultados dos estimadores avaliados e retorna o mesmo DataFrame exibido para realizar
        processamentos necessários.

        :return: DataFrame com os dados dos processamentos dos Pipelines.
        """

        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by=['mean', 'median', 'standard_deviation'], ascending=False)

        print(tabulate(df_results, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

        return df_results

    @staticmethod
    def _format_time(seconds):
        """
        Função para formatar os segundos do processamento em um texto mais legível no estilo: 00:00:00.000
        """

        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"


class ScikitLearnMultiProcessManager(MultiProcessManager):
    """
    Implementação de MultiProcessManager específica para avaliar estimadores da biblioteca ScikitLearn.
    """

    def __init__(self,
                 data_x,
                 data_y, seed: int,
                 fold_splits: int,
                 pipelines: list[ScikitLearnPipeline] | ScikitLearnPipeline,
                 history_manager: HistoryManager,
                 stratified: bool = False,
                 scoring: str = 'neg_mean_squared_error',
                 save_history: bool = True,
                 history_index: int = None):
        super().__init__(data_x, data_y, seed, fold_splits, pipelines, history_manager, stratified, scoring,
                         save_history, history_index)

    def _process_validation(self, pipeline: ScikitLearnPipeline, search_cv: RandomizedSearchCV) -> ValidationResult:
        if search_cv is None:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)
        else:
            return pipeline.validator.validate(searcher=search_cv,
                                               data_x=self.data_x,
                                               data_y=self.data_y,
                                               scoring=self.scoring,
                                               cv=self.cv)

    def _on_after_process_pipelines(self, df_results: DataFrame):
        self.__save_best_estimator(df_results)

    def __save_best_estimator(self, df_results: DataFrame):
        """
        Função para salvar o melhor estimador entre os melhores encontrados em cada pipeline. Isso pode ser salvo em um
        arquivo separado pois utiliza o history_manager do ProcessManager e não do pipeline.
        """

        def is_best_pipeline(pipe):
            return (
                    best['estimator'].values[0] == type(pipe.estimator).__name__ and
                    best['feature_searcher'].values[0] == type(pipe.feature_searcher).__name__ and
                    best['params_searcher'].values[0] == type(pipe.params_searcher).__name__ and
                    best['validator'].values[0] == type(pipe.validator).__name__
            )

        if self.save_history and self.history_index is None:
            best = df_results.head(1)

            best_pipeline = [pipe for pipe in self.pipelines if is_best_pipeline(pipe)][0]
            validation_result = best_pipeline.history_manager.load_validation_result_from_history()
            dict_history = best_pipeline.history_manager.get_dictionary_from_json(index=-1)

            self.history_manager.save_result(classifier_result=validation_result,
                                             feature_selection_time=best['feature_selection_time'].values[0],
                                             search_time=best['search_time'].values[0],
                                             validation_time=best['validation_time'].values[0],
                                             scoring=best['scoring'].values[0],
                                             features=dict_history['features'].split(','))


class XGBoostMultiProcessManager(MultiProcessManager):
    """
    Implementação de MultiProcessManager específica para avaliar estimadores da biblioteca XGBoost.
    """

    def __init__(self,
                 data_x,
                 data_y,
                 seed: int,
                 fold_splits: int,
                 pipelines: list[XGBoostPipeline] | XGBoostPipeline,
                 history_manager: HistoryManager,
                 stratified: bool = False,
                 scoring: str = 'neg_mean_squared_error',
                 save_history: bool = True,
                 history_index: int = None):
        super().__init__(data_x, data_y, seed, fold_splits, pipelines, history_manager, stratified, scoring,
                         save_history, history_index)

    def _process_validation(self, pipeline: XGBoostPipeline, search_cv: RandomizedSearchCV) -> ValidationResult:
        if search_cv is None:
            return pipeline.history_manager.load_validation_result_from_history(self.history_index)
        else:
            return pipeline.validator.validate(
                searcher=search_cv,
                train_matrix=xgb.DMatrix(self.data_x, self.data_y),
                cv=self.cv)
