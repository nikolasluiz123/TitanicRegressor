from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from tabulate import tabulate


class ValidationResult(ABC):

    @abstractmethod
    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        ...


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

    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        pipeline_infos['mean'] = self.mean
        pipeline_infos['standard_deviation'] = self.standard_deviation
        pipeline_infos['median'] = self.median
        pipeline_infos['variance'] = self.variance
        pipeline_infos['standard_error'] = self.standard_error
        pipeline_infos['min_max_score'] = self.min_max_score
        pipeline_infos['scoring'] = self.scoring

        return pipeline_infos


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

    def append_data(self, pipeline_infos: dict[str, Any]) -> dict[str, Any]:
        pipeline_infos['metrics'] = self.metrics
        pipeline_infos['train_means'] = self.train_means
        pipeline_infos['train_standard_errors'] = self.train_standard_errors
        pipeline_infos['test_means'] = self.test_means
        pipeline_infos['test_standard_errors'] = self.test_standard_errors

        return pipeline_infos