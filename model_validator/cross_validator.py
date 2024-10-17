import time

import numpy as np
from sklearn.model_selection import cross_val_score

from model_validator.result import CrossValidationResult
from model_validator.validator import BaseValidator


class CrossValidator(BaseValidator):
    """
    Classe que implementa a validação crusada do modelo encontrado pela busca de hiper parâmetros
    """

    def __init__(self,
                 log_level: int = 1,
                 n_jobs: int = -1):
        super().__init__(log_level, n_jobs)

    def validate(self, searcher, data_x, data_y, scoring='neg_mean_squared_error') -> CrossValidationResult:
        self.start_best_model_validation = time.time()

        scores = cross_val_score(estimator=searcher,
                                 X=data_x,
                                 y=data_y,
                                 cv=self.cv,
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
            min_max_score=(np.min(scores), np.max(scores)),
            estimator=searcher.best_estimator_,
            scoring=scoring
        )

        return result
