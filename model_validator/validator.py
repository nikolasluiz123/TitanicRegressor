from abc import ABC, abstractmethod

from sklearn.model_selection import KFold

from model_validator.result import CrossValidationResult


class BaseValidator(ABC):
    """
    Classe base que todos os validadores de modelo devem implementar
    """

    def __init__(self,
                 log_level: int = 1,
                 n_jobs: int = -1):
        self.log_level = log_level
        self.n_jobs = n_jobs

        self.cv = KFold(n_splits=5, shuffle=True)
        self.start_best_model_validation = 0
        self.end_best_model_validation = 0

    @abstractmethod
    def validate(self, searcher, data_x, data_y, scoring='neg_mean_squared_error') -> CrossValidationResult:
        """
        Função para realizar a validação do modelo utilizando alguma estratégia.

        :return: Retorna um objeto CrossValScoreResult contendo as métricas matemáticas
        """
