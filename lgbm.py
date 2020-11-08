import joblib

from filepreprocessing import preprocessing_lgb, ratings_to_numbers, result_funct, function_cat
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import cross_val_score

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train, test = preprocessing_lgb(train, test)


def tune_params(train):
    x = train.drop(columns=['Id', 'SalePrice'])
    y = train['SalePrice']

    def objective(trial):
        lgb_max_depth = trial.suggest_int('lgb_max_depth', 12, 60)
        lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 500, 1000)
        lgb_learning_rate = trial.suggest_uniform('lgb_learning_rate', 0.01, 0.15)
        lgb_subsample = trial.suggest_uniform('lgb_subsample', 0.5, 1)

        regressor_obj = lgb.LGBMRegressor(max_depth=lgb_max_depth,
                                          n_estimators=lgb_n_estimators,
                                          learning_rate=lgb_learning_rate,
                                          subsample=lgb_subsample,
                                          random_state=5)

        score = cross_val_score(regressor_obj, x, y, n_jobs=-1, cv=4, scoring='neg_mean_squared_error')
        score_mean = score.mean()
        return score_mean

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)
    params = study.best_params

    joblib.dump(params, 'best_optuna_params.pkl')

    return params


params = tune_params(train)

model = lgb.LGBMRegressor(max_depth=params['lgb_max_depth'], subsample=params['lgb_subsample'],
                          learning_rate=params['lgb_learning_rate'], n_estimators=params['lgb_n_estimators'])

model.fit(train.drop(columns=['Id', 'SalePrice']),
          train['SalePrice'], categorical_feature=function_cat(train))
joblib.dump(model, 'LGBM.pkl')

test['SalePrice'] = model.predict(test.drop(columns=['Id']))

result_funct(test, 'LGB')
