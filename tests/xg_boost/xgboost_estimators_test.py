import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform

from data.data_processing import get_train_data
from hiper_params_search.random_searcher import RandomHipperParamsSearcher
from manager.history_manager import XGBoostValidationHistoryManager
from manager.multi_process_manager import XGBoostMultiProcessManager
from manager.multi_process_manager_pipelines import XGBoostPipeline
from model_validator.cross_validator import XGBoostCrossValidator
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

xgboost_regressor_history_manager = XGBoostValidationHistoryManager(output_directory='history',
                                                                    models_directory='regressor_models',
                                                                    params_file_name='regressor_best_params')

best_model_history_manager = XGBoostValidationHistoryManager(output_directory='history',
                                                             models_directory='models',
                                                             params_file_name='best_params')
manager = XGBoostMultiProcessManager(
    data_x=x,
    data_y=y,
    seed=42,
    fold_splits=10,
    scoring='neg_mean_squared_error',
    pipelines=XGBoostPipeline(
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
        history_manager=xgboost_regressor_history_manager
    ),
    history_manager=best_model_history_manager,
    save_history=True,
    history_index=0,
)

manager.process_pipelines()