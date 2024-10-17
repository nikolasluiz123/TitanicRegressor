import json
import os
import pickle
from abc import ABC, abstractmethod

from model_validator.result import CrossValidationResult, ValidationResult


class HistoryManager(ABC):
    """
    Classe responsável por armazenar os dados históricos das buscas de hiper parâmetros dos modelos. Isso pode evitar
    um reprocessamento quando desejar apenas exibir novamente no console um resultado obtido em uma das tentativas.

    Os dados referentes ao desempenho do modelo são salvos no formato de JSON, dentro de uma lista, onde poderão ser
    recuperados através do seu índice. Além disso, o próprio modelo é salvo para que possa ser utilizado com dados diferentes
    e possa ser verificado o seu comportamento.
    """

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str):
        """
        :param output_directory: Diretório que armazenará todos os dados de histórico
        :param models_directory: Diretório específico para os modelos treinados
        :param params_file_name: Nome do arquivo JSON que salvará os valores dos parâmetros que geraram o melhor modelo
        """
        self.output_directory = output_directory
        self.models_directory = os.path.join(self.output_directory, models_directory)
        self.params_file_name = params_file_name

    @abstractmethod
    def save_result(self,
                    classifier_result,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str]):
        """
        Função responsável por salvar todos os dados relevantes para o histórico.

        :param classifier_result: Resultado da validação
        :param search_time: Tempo que levou a execução da busca por parâmetros
        :param validation_time: Tempo que levou a execução da validação
        """

    def _create_output_dir(self):
        """
        Função para criar o diretório de histório caso não exista. É nesse diretório que o arquivo JSON e os modelos
        ficarão.
        """
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    def _save_dictionary_in_json(self, dictionary):
        """
        Função utilizada para adicionar o dicionário com os valores resultantes da busca dentro da lista do JSON

        :param dictionary: Dicionário com os dados
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                data = json.load(file)
        else:
            data = []

        data.append(dictionary)

        with open(output_path, 'w') as file:
            json.dump(data, file, indent=4)

    def has_history(self) -> bool:
        """
        Retorna se há ao menos um registro dentro do arquivo de histórico
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return False

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data) > 0

    @abstractmethod
    def load_result_from_history(self, index: int = -1) -> ValidationResult:
        """
        Função para recuperar um registro da lista do JSON de histórico

        :param index: Índice da lista que deseja retornar
        """

    def _get_dictionary_from_json(self, index):
        """
        Retorna um dicionário a partir do JSON do histórico

        :param index: Índice da lista de histórico que deseja recuperar
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            raise FileNotFoundError(
                f"O arquivo {self.params_file_name}.json não foi encontrado no diretório {self.output_directory}.")

        with open(output_path, 'r') as file:
            data = json.load(file)

        if index < -1 or index >= len(data):
            raise IndexError(f"Índice {index} fora dos limites. O arquivo contém {len(data)} entradas.")

        result_dict = data[index]

        return result_dict

    def _save_model(self, estimator):
        """
        Função para salvar o modelo treinado e utilizá-lo para prever com outros dados.

        :param estimator: Estimador que deseja salvar
        """

        history_len = self._get_history_len()
        output_path = os.path.join(self.models_directory, f"model_{history_len}.pkl")

        with open(output_path, 'wb') as file:
            pickle.dump(estimator, file)

    def get_saved_model(self, version: int):
        """
        Recupera o modelo que foi salvo de acordo com a versão

        :param version: Versão do modelo, concatenada no nome do arquivo, que deseja recuperar.
        """

        output_path = os.path.join(self.models_directory, f"model_{version}.pkl")

        with open(output_path, 'rb') as f:
            return pickle.load(f)

    def _get_history_len(self) -> int:
        """
        Retorna o tamanho da lista do histórico
        """
        output_path = os.path.join(self.output_directory, f"{self.params_file_name}.json")

        if not os.path.exists(output_path):
            return 0

        with open(output_path, 'r') as file:
            data = json.load(file)
            return len(data)


class CrossValidationHistoryManager(HistoryManager):
    """
    Classe para manipular o histórico quando é utilizada a estratégia de validação cruzada, que gera N valores e podem
    ser calculadas diversas métricas e obter resultados confiáveis.
    """

    def __init__(self, output_directory: str, models_directory: str, params_file_name: str):
        super().__init__(output_directory, models_directory, params_file_name)

    def save_result(self,
                    classifier_result: CrossValidationResult,
                    feature_selection_time: str,
                    search_time: str,
                    validation_time: str,
                    scoring: str,
                    features: list[str]):
        dictionary = {
            'mean': classifier_result.mean,
            'standard_deviation': classifier_result.standard_deviation,
            'median': classifier_result.median,
            'variance': classifier_result.variance,
            'standard_error': classifier_result.standard_error,
            'min_max_score': classifier_result.min_max_score,
            'estimator_params': classifier_result.estimator.get_params(),
            'scoring': scoring,
            'features': ", ".join(features),
            'feature_selection_time': feature_selection_time,
            'search_time': search_time,
            'validation_time': validation_time
        }

        self._create_output_dir()
        self._save_dictionary_in_json(dictionary)
        self._save_model(classifier_result.estimator)

    def load_result_from_history(self, index: int = -1) -> CrossValidationResult:
        result_dict = self._get_dictionary_from_json(index)

        return CrossValidationResult(
            mean=result_dict['mean'],
            standard_deviation=result_dict['standard_deviation'],
            median=result_dict['median'],
            variance=result_dict['variance'],
            standard_error=result_dict['standard_error'],
            min_max_score=result_dict['min_max_score'],
            scoring=result_dict['scoring'],
            estimator=self.get_saved_model(self._get_history_len()),
        )
