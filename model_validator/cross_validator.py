import time

import numpy as np
import xgboost as xgb
from pandas import DataFrame
from sklearn.model_selection import cross_val_score, KFold
from xgboost import DMatrix

from model_validator.result import CrossValidationResult, XGBoostCrossValidationResult
from model_validator.validator import ScikitLearnBaseValidator, XGBoostBaseValidator, XGBoostCrossValidationMetrics


class CrossValidatorScikitLearn(ScikitLearnBaseValidator):
    """
    Classe que implementa a validação crusada do modelo encontrado pela busca de hiper parâmetros
    """

    def __init__(self,
                 log_level: int = 0,
                 n_jobs: int = -1):
        super().__init__(log_level, n_jobs)

    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 scoring='neg_mean_squared_error',
                 cv=KFold(n_splits=5, shuffle=True)) -> CrossValidationResult:
        self.start_best_model_validation = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=data_x,
                                 y=data_y,
                                 cv=cv,
                                 n_jobs=self.n_jobs,
                                 verbose=self.log_level,
                                 scoring=scoring)

        self.end_best_model_validation = time.time()

        result = CrossValidationResult(
            mean=np.mean(scores),
            standard_deviation=np.std(scores),
            median=np.median(scores),
            variance=np.var(scores),
            standard_error=np.std(scores) / np.sqrt(len(scores)),
            min_max_score=(round(float(np.min(scores)), 4), round(float(np.max(scores)), 4)),
            estimator=searcher.best_estimator_,
            scoring=scoring
        )

        return result


class XGBoostCrossValidator(XGBoostBaseValidator):

    def __init__(self,
                 interation_number: int,
                 metrics: list[XGBoostCrossValidationMetrics],
                 early_stopping_rounds: int,
                 verbose_eval: int):
        super().__init__(interation_number, metrics, early_stopping_rounds, verbose_eval)

    def validate(self,
                 searcher,
                 train_matrix: DMatrix,
                 cv) -> XGBoostCrossValidationResult:
        metrics_ = [m.value for m in self.metrics]

        data_frame = xgb.cv(
            dtrain=train_matrix,
            params=searcher.best_params_,
            num_boost_round=self.interation_number,
            folds=cv,
            metrics=metrics_,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval
        )

        return self.__extract_results(searcher=searcher, cv_df=data_frame, metrics=metrics_)

    @staticmethod
    def __extract_results(searcher, cv_df: DataFrame, metrics: list[str]) -> XGBoostCrossValidationResult:
        def __get_last_row_value(col_name: str) -> float:
            return round(float(cv_df[col_name].iloc[-1] if col_name in cv_df.columns else float('nan')), 4)

        train_means, train_std_errs = [], []
        test_means, test_std_errs = [], []

        for metric in metrics:
            train_mean = __get_last_row_value(f'train-{metric}-mean')
            train_std = __get_last_row_value(f'train-{metric}-std')
            test_mean = __get_last_row_value(f'test-{metric}-mean')
            test_std = __get_last_row_value(f'test-{metric}-std')

            train_means.append((metric, train_mean))
            train_std_errs.append((metric, train_std))
            test_means.append((metric, test_mean))
            test_std_errs.append((metric, test_std))

        return XGBoostCrossValidationResult(
            train_means=train_means,
            train_standard_errors=train_std_errs,
            test_means=test_means,
            test_standard_errors=test_std_errs,
            metrics=metrics,
            best_params=searcher.best_params_,
            estimator=searcher.best_estimator_
        )
