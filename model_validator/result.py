from abc import ABC, abstractmethod

import pandas as pd
from tabulate import tabulate


class ValidationResult(ABC):
    """
    Classe base de resultado de validação dos modelos de classificação
    """

    @abstractmethod
    def show_cross_val_metrics(self):
        """
        Função para exibir as métricas de validação.
        """


class CrossValidationResult(ValidationResult):

    def __init__(self,
                 mean: float,
                 standard_deviation: float,
                 median: float,
                 variance: float,
                 standard_error: float,
                 min_max_score: tuple[float, float],
                 estimator,
                 scoring: str):
        """
            :param mean: Média dos scores individuais, fornece uma estimativa central do desempenho do modelo.
            :param standard_deviation: Desvio Padrão, mede a variação dos scores em diferentes folds. Um Desvio Padrão
            baixo indica que o modelo tem desempenho consistente, enquanto um desvio padrão alto indica variabilidade
            entre os folds.
            :param median: A mediana dos scores é uma métrica robusta que representa o valor central da distribuição dos
            scores, sendo menos sensível a outliers.
            :param variance: A variância mede a dispersão dos scores e está relacionada ao desvio padrão, sendo o
            quadrado deste.
            :param standard_error: O erro padrão da média estima a precisão da média dos scores, mostrando o quão longe
            a média estimada está da média verdadeira.
            :param min_max_score: O score máximo e mínimo ajudam a identificar a melhor e a pior performance entre os
            folds.
            :param estimator Estimador com os melhores parâmetros e que foi testado.
        """

        self.mean = mean
        self.standard_deviation = standard_deviation
        self.median = median
        self.variance = variance
        self.standard_error = standard_error
        self.min_max_score = min_max_score
        self.estimator = estimator
        self.scoring = scoring

    def show_cross_val_metrics(self):
        print("Resultados das Métricas de Validação Cruzada da Classificação")
        print("-" * 50)
        print(f"Média dos scores          : {self.mean:.4f}")
        print(f"Desvio padrão             : {self.standard_deviation:.4f}")
        print(f"Mediana dos scores        : {self.median:.4f}")
        print(f"Variância dos scores      : {self.variance:.4f}")
        print(f"Erro padrão da média      : {self.standard_error:.4f}")
        print(f"Score mínimo              : {self.min_max_score[0]:.4f}")
        print(f"Score máximo              : {self.min_max_score[1]:.4f}")
        print(f"Scoring                   : {self.scoring}")
        print(f"Melhor Estimator          : {self.estimator} ")
        print("-" * 50)


class XGBoostCrossValidationResult(ValidationResult):

    def __init__(self,
                 train_means: list[tuple[str, float]],
                 train_standard_errors: list[tuple[str, float]],
                 test_means: list[tuple[str, float]],
                 test_standard_errors: list[tuple[str, float]],
                 metrics: list[str],
                 best_params,
                 estimator):

        self.train_means = train_means
        self.train_standard_errors = train_standard_errors
        self.test_means = test_means
        self.test_standard_errors = test_standard_errors
        self.metrics = metrics
        self.best_params = best_params
        self.estimator = estimator

    def show_cross_val_metrics(self):
        data = {
            "Métricas": self.metrics,
            "Média no Treino": [self._find_value(self.train_means, m) for m in self.metrics],
            "Desvio Padrão no Treino": [self._find_value(self.train_standard_errors, m) for m in self.metrics],
            "Média no Teste": [self._find_value(self.test_means, m) for m in self.metrics],
            "Desvio Padrão no Teste": [self._find_value(self.test_standard_errors, m) for m in self.metrics],
        }
        df = pd.DataFrame(data)

        print('\nResultados das Métricas de Validação Cruzada')
        print(tabulate(df, headers='keys', tablefmt='fancy_grid', floatfmt=".6f"))

    def _find_value(self, data: list[tuple[str, float]], metric: str) -> float:
        """Função auxiliar para encontrar o valor correspondente a uma métrica."""

        for m, value in data:
            if m == metric:
                return value

        return float('nan')
