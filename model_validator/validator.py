from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pandas import DataFrame
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import DMatrix

from model_validator.result import CrossValidationResult, XGBoostCrossValidationResult


class ScikitLearnBaseValidator(ABC):
    """
    Classe base que todos os validadores de modelo devem implementar
    """

    def __init__(self,
                 log_level: int = 1,
                 n_jobs: int = -1):
        self.log_level = log_level
        self.n_jobs = n_jobs

        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self,
                 searcher,
                 data_x,
                 data_y,
                 scoring='neg_mean_squared_error',
                 cv=KFold(n_splits=5, shuffle=True)) -> CrossValidationResult:
        """
        Função para realizar a validação do modelo utilizando alguma estratégia.

        :return: Retorna um objeto CrossValScoreResult contendo as métricas matemáticas
        """


class XGBoostCrossValidationMetrics(Enum):
    ERROR = 'error'
    RMSE = 'rmse'
    LOG_LOSS = 'logloss'
    MAE = 'mae'
    MAPE = 'mape'
    AUC = 'auc'
    AUC_PR = 'aucpr'
    M_ERROR = 'merror'


class XGBoostBaseValidator(ABC):

    def __init__(self,
                 interation_number: int,
                 metrics: list[XGBoostCrossValidationMetrics],
                 early_stopping_rounds: int,
                 verbose_eval: int):
        self.interation_number = interation_number
        self.metrics = metrics
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval

        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self,
                 searcher,
                 train_matrix: DMatrix,
                 cv) -> XGBoostCrossValidationResult:
        ...
