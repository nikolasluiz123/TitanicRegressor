import pandas as pd
from scipy.stats import randint, uniform
from sklearn.linear_model import SGDRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import RegressorRandomHipperParamsSearcher
from manager.history_manager import CrossValidationHistoryManager
from manager.process_manager import ProcessManager
from model_validator.cross_validator import CrossValidator
from regression_vars_search.k_best_feature_searcher import SelectKBestFeatureSearcher
from regression_vars_search.recursive_feature_searcher import RecursiveFeatureSearcher

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns
x = pd.get_dummies(x, columns=obj_columns)

print()
print(f'Todas as Features')
print(tabulate(x.head(), headers='keys', tablefmt='psql', showindex=False))
print()

y = df_train['sobreviveu']

search_params = {
    'max_iter': randint(100, 2000),
    'fit_intercept': [True, False]
}

feature_searcher = SelectKBestFeatureSearcher(log_level=0, feature_number=3)
params_searcher = RegressorRandomHipperParamsSearcher(params=search_params)
validator = CrossValidator()
history_manager = CrossValidationHistoryManager(output_directory='history',
                                                models_directory='models_select_k_best',
                                                params_file_name='tested_params_select_k_best')

process_manager = ProcessManager(
    data_x=x,
    data_y=y,
    estimator=TheilSenRegressor(),
    seed=42,
    feature_searcher=feature_searcher,
    params_searcher=params_searcher,
    validator=validator,
    history_manager=history_manager,
    save_history=True,
)

process_manager.process(number_interations=10)