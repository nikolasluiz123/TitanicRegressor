import pandas as pd
import xgboost as xgb
from scipy.stats import uniform, randint
from tabulate import tabulate

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import RegressorRandomHipperParamsSearcher
from manager.history_manager import XGBoostValidationHistoryManager
from manager.process_manager import XGBoostProcessManager
from model_validator.cross_validator import XGBoostCrossValidator
from model_validator.validator import XGBoostCrossValidationMetrics
from regression_vars_search.k_best_feature_searcher import SelectKBestFeatureSearcher

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
    'colsample_bytree': uniform(loc=0.1, scale=0.8),
    'subsample': uniform(loc=0.1, scale=0.8),
    'max_depth': randint(1, 20),
    'learning_rate': uniform(loc=0.01, scale=1),
    'gamma': uniform(loc=0, scale=0.9),
    'min_child_weight': randint(1, 10),
    'max_delta_step': randint(0, 10)
}

feature_searcher = SelectKBestFeatureSearcher(log_level=1, feature_number=5)
params_searcher = RegressorRandomHipperParamsSearcher(number_iterations = 100, params=search_params, log_level=1)
validator = XGBoostCrossValidator(interation_number=100,
                                  metrics=[XGBoostCrossValidationMetrics.RMSE],
                                  early_stopping_rounds=1,
                                  verbose_eval=5)

history_manager = XGBoostValidationHistoryManager(output_directory='history',
                                                  models_directory='models_rfe_cv',
                                                  params_file_name='tested_params_rfe_cv')

process_manager = XGBoostProcessManager(
    data_x=x,
    data_y=y,
    estimator=xgb.XGBRegressor(),
    seed=42,
    scoring='neg_mean_squared_error',
    feature_searcher=feature_searcher,
    params_searcher=params_searcher,
    validator=validator,
    history_manager=history_manager,
    save_history=False,
)

process_manager.process()
