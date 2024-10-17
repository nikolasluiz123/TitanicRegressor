from abc import ABC, abstractmethod


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
