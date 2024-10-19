import pandas as pd
import xgboost as xgb

from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import RandomHipperParamsSearcher
from manager.history_manager import CrossValidationHistoryManager, XGBoostValidationHistoryManager
from manager.sk_learn_multi_process_manager import ScikitLearnPipeline, ScikitLearnMultiProcessManager
from manager.xg_boost_multi_process_manager import XGBoostPipeline, XGBoostMultiProcessManager
from model_validator.cross_validator import CrossValidatorScikitLearn, XGBoostCrossValidator
from model_validator.validator import XGBoostCrossValidationMetrics
from regression_vars_search.k_best_feature_searcher import SelectKBestFeatureSearcher

df_train = get_train_data()

x = df_train.drop(columns=['sobreviveu'], axis=1)

obj_columns = df_train.select_dtypes(include='object').columns

x = pd.get_dummies(x, columns=obj_columns)
y = df_train['sobreviveu']

feature_searcher = SelectKBestFeatureSearcher(feature_number=5, log_level=1)
params_searcher = RandomHipperParamsSearcher(number_iterations=100, log_level=1)
cross_validator = XGBoostCrossValidator(interation_number=100,
                                        metrics=[XGBoostCrossValidationMetrics.RMSE],
                                        early_stopping_rounds=1,
                                        verbose_eval=5)

pipelines = [
    XGBoostPipeline(
        estimator=xgb.XGBRegressor(),
        params={
            'colsample_bytree': uniform(loc=0.1, scale=0.8),
            'subsample': uniform(loc=0.1, scale=0.8),
            'max_depth': randint(1, 20),
            'learning_rate': uniform(loc=0.01, scale=1),
            'gamma': uniform(loc=0, scale=0.9),
            'min_child_weight': randint(1, 10),
            'max_delta_step': randint(0, 10)
        },
        feature_searcher=feature_searcher,
        params_searcher=params_searcher,
        validator=cross_validator,
        history_manager=XGBoostValidationHistoryManager(
            output_directory='history',
            models_directory='regressor_models',
            params_file_name='regressor_best_params')
    )
]

manager = XGBoostMultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    fold_splits=10,
    pipelines=pipelines,
    history_manager=CrossValidationHistoryManager(
        output_directory='history',
        models_directory='models',
        params_file_name='best_params'
    ),
    save_history=True
)

manager.process_pipelines()