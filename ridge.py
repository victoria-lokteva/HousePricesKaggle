import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

from elastic_net import feature_selection
from filepreprocessing import preprocessing, result_funct

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train, test = preprocessing(train, test)


def tune_regression_params(train):
    scores = dict()
    for alpha in tqdm(np.linspace(0, 1, 50)):
        for normalize in [True, False]:
            reg_ridge = Ridge(alpha=alpha, normalize=normalize)
            x, y = train.drop(columns=['Id', 'SalePrice']), train['SalePrice']
            score = cross_val_score(reg_ridge, x, y, scoring='neg_mean_squared_error', cv=4)
            scores[(alpha, normalize)] = np.mean(score)
    opt_params = max(scores, key=scores.get)
    return opt_params


for i in range(0, 2):
    opt_params = tune_regression_params(train)
    reg_ridge_opt = Ridge(alpha=opt_params[0], normalize=opt_params[1])
    bad_features = feature_selection(reg_ridge_opt, train)
    train = train.drop(bad_features)

opt_params = tune_regression_params(train)
reg_ridge_opt = Ridge(alpha=opt_params[0], normalize=opt_params[1])

reg_ridge_opt.fit(train.drop(columns=['SalePrice', 'Id']), train['SalePrice'])

joblib.dump(reg_ridge_opt, 'Model_Reg_Ridge_opt.pkl')

test['SalePrice'] = reg_ridge_opt.predict(test.drop(columns=['Id']))

result_funct(test, 'Ridge')
