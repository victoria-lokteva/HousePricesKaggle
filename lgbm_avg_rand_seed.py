import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from filepreprocessing import preprocessing_lgb, result_funct, function_cat

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train, test = preprocessing_lgb(train, test)

params = joblib.load('best_optuna_params.pkl')

x_test = test.drop(columns=['Id'])
model_list = []
y_pred = []

x = train.drop(columns=['Id', 'SalePrice'])
y = train['SalePrice']
for i in range(3, 103, 5):
    regressor_obj = lgb.LGBMRegressor(max_depth=params['lgb_max_depth'],
                                      subsample=params['lgb_subsample'],
                                      learning_rate=params['lgb_learning_rate'],
                                      n_estimators=params['lgb_n_estimators'],
                                      random_state=i)

    regressor_obj.fit(x, y, categorical_feature=function_cat(train))

    model_list.append(regressor_obj)

    y_pred.append(regressor_obj.predict(x_test))

test['SalePrice'] = np.mean(y_pred, axis=0)

test = result_funct(test, 'LGB+ mean random seed')

