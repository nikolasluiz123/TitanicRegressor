import warnings

import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import RandomHipperParamsSearcher
from manager.history_manager import CrossValidationHistoryManager
from manager.multi_process_manager import ScikitLearnPipeline, ScikitLearnMultiProcessManager
from model_validator.cross_validator import CrossValidatorScikitLearn
from regression_vars_search.recursive_feature_searcher import RecursiveFeatureSearcher

warnings.filterwarnings("ignore", category=RuntimeWarning)

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns

x = pd.get_dummies(x, columns=obj_columns)
y = df_train['sobreviveu']

feature_searcher = RecursiveFeatureSearcher(log_level=1)
params_searcher = RandomHipperParamsSearcher(number_iterations=1000, log_level=1)
cross_validator = CrossValidatorScikitLearn(log_level=1)

pipelines = [
    ScikitLearnPipeline(
        estimator=DecisionTreeRegressor(),
        params={
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'splitter': ['best', 'random'],
            'max_depth': randint(1, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2'],
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='decision_tree_regressor_models',
            params_file_name='decision_tree_regressor_best_params')
    ),
    ScikitLearnPipeline(
        estimator=ElasticNet(),
        params={
            'alpha': uniform(loc=0.1, scale=0.9),
            'l1_ratio': uniform(loc=0.1, scale=0.9),
            'fit_intercept': [True, False],
            'max_iter': randint(100, 2000),
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='elastic_net_models',
            params_file_name='elastic_net_best_params')
    ),
    ScikitLearnPipeline(
        estimator=GradientBoostingRegressor(),
        params={
            'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
            'learning_rate': uniform(loc=0.01, scale=0.99),
            'n_estimators': randint(100, 300),
            'subsample': uniform(loc=0.1, scale=0.9),
            'criterion': ['friedman_mse', 'squared_error'],
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_depth': randint(1, 20),
            'max_features': [None, 'sqrt', 'log2']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='gradient_boosting_regressor_models',
            params_file_name='gradient_boosting_regressor_best_params')
    ),
    ScikitLearnPipeline(
        estimator=RandomForestRegressor(),
        params={
            'n_estimators': randint(10, 50),
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': randint(1, 20),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 20),
            'min_weight_fraction_leaf': uniform(loc=0.1, scale=0.4),
            'max_features': [None, 'sqrt', 'log2']
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='random_forest_regressor_models',
            params_file_name='random_forest_regressor_best_params')
    ),
    ScikitLearnPipeline(
        estimator=SGDRegressor(),
        params={
            'loss': ['huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'alpha': uniform(loc=0.1, scale=0.9),
            'max_iter': randint(100, 2000),
            'l1_ratio': uniform(loc=0.1, scale=0.9),
            'fit_intercept': [True, False]
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='sgd_regressor_models',
            params_file_name='sgd_regressor_best_params')
    ),
    ScikitLearnPipeline(
        estimator=TheilSenRegressor(),
        params={
            'max_iter': randint(100, 2000),
            'fit_intercept': [True, False]
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=CrossValidationHistoryManager(
            output_directory='history',
            models_directory='theil_sen_regressor_models',
            params_file_name='theil_sen_regressor_best_params')
    )
]

best_params_history_manager = CrossValidationHistoryManager(output_directory='history_bests',
                                                            models_directory='best_models',
                                                            params_file_name='best_params')
manager = ScikitLearnMultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    fold_splits=10,
    pipelines=pipelines,
    history_manager=best_params_history_manager,
    save_history=True,
    history_index=-1
)

manager.process_pipelines()