import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Any
from sklearn.model_selection import KFold, StratifiedKFold, RandomizedSearchCV
from tabulate import tabulate

from hiper_params_search.random_searcher import RandomHipperParamsSearcher
from manager.history_manager import HistoryManager
from model_validator.result import ValidationResult
from model_validator.validator import ScikitLearnBaseValidator, XGBoostBaseValidator
from regression_vars_search.features_searcher import FeaturesSearcher


class XGBoostPipeline:

    def __init__(self,
                 estimator,
                 params,
                 feature_searcher: FeaturesSearcher,
                 params_searcher: RandomHipperParamsSearcher,
                 validator: XGBoostBaseValidator,
                 history_manager: HistoryManager):
        self.estimator = estimator
        self.params = params
        self.feature_searcher = feature_searcher
        self.params_searcher = params_searcher
        self.validator = validator
        self.history_manager = history_manager

    def get_dict_pipeline_data(self) -> dict[str, Any]:
        return {
            'estimator': type(self.estimator).__name__,
            'feature_searcher': type(self.feature_searcher).__name__,
            'params_searcher': type(self.params_searcher).__name__,
            'validator': type(self.validator).__name__,
            'history_manager': type(self.history_manager).__name__
        }


class XGBoostMultiProcessManager:

    def __init__(self,
                 data_x,
                 data_y,
                 seed: int,
                 fold_splits: int,
                 pipelines: list[XGBoostPipeline],
                 history_manager: HistoryManager,
                 stratified: bool = False,
                 scoring: str = 'neg_mean_squared_error',
                 save_history: bool = True,
                 history_index: int = None):
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

    def process_pipelines(self):
        for pipeline in self.pipelines:
            self.__process_feature_selection(pipeline)
            search_cv = self.__process_hiper_params_search(pipeline)
            validation_result = self.__process_validation(pipeline, search_cv)

            self.__save_data_in_history(pipeline, validation_result)
            self.__append_new_result(pipeline, validation_result)

        self.__show_results()

    def __process_feature_selection(self, pipeline: XGBoostPipeline):
        self.data_x = pipeline.feature_searcher.select_features(
            estimator=pipeline.estimator,
            data_x=self.data_x,
            data_y=self.data_y,
            scoring=self.scoring,
            cv=self.cv
        )

    def __process_hiper_params_search(self, pipeline: XGBoostPipeline) -> RandomizedSearchCV | None:
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

    def __process_validation(self, pipeline: XGBoostPipeline, search_cv: RandomizedSearchCV) -> ValidationResult:
        if search_cv is None:
            return pipeline.history_manager.load_result_from_history(self.history_index)
        else:
            return pipeline.validator.validate(
                searcher=search_cv,
                train_matrix=xgb.DMatrix(self.data_x, self.data_y),
                cv=self.cv
            )

    def __save_data_in_history(self, pipeline: XGBoostPipeline, result: ValidationResult):
        if self.save_history and self.history_index is None:
            feature_selection_time = pipeline.feature_searcher.end_search_features_time - pipeline.feature_searcher.start_search_features_time
            search_time = pipeline.params_searcher.end_search_parameter_time - pipeline.params_searcher.start_search_parameter_time
            validation_time = pipeline.validator.end_best_model_validation - pipeline.validator.start_best_model_validation

            pipeline.history_manager.save_result(result,
                                                 feature_selection_time=self.__format_time(feature_selection_time),
                                                 search_time=self.__format_time(search_time),
                                                 validation_time=self.__format_time(validation_time),
                                                 scoring=self.scoring,
                                                 features=self.data_x.columns.tolist())

    def __append_new_result(self, pipeline: XGBoostPipeline, result: ValidationResult):
        pipeline_infos = pipeline.get_dict_pipeline_data()
        performance_metrics = result.append_data(pipeline_infos)

        feature_selection_time = pipeline.feature_searcher.end_search_features_time - pipeline.feature_searcher.start_search_features_time
        search_time = pipeline.params_searcher.end_search_parameter_time - pipeline.params_searcher.start_search_parameter_time
        validation_time = pipeline.validator.end_best_model_validation - pipeline.validator.start_best_model_validation

        performance_metrics['feature_selection_time'] = self.__format_time(feature_selection_time)
        performance_metrics['search_time'] = self.__format_time(search_time)
        performance_metrics['validation_time'] = self.__format_time(validation_time)

        self.results.append(performance_metrics)

    def __show_results(self):
        df = pd.DataFrame(self.results)
        df = df.sort_values(by='test_means', ascending=True)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".6f", showindex=False))

    @staticmethod
    def __format_time(seconds):
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{milliseconds:03}"
