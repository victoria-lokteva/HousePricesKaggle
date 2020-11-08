from filepreprocessing import preprocessing, result_funct
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
import joblib
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


def tune_regression_params(train):
    x, y = train.drop(columns=['Id', 'SalePrice']), train['SalePrice']

    scores = dict()
    for alpha in np.linspace(0, 0.5, 15):
        for normalize in [True, False]:
            for l1_ratio in np.linspace(0, 0.5, 6):
                regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, normalize=normalize)
                regr.fit(x, y)
                score = cross_val_score(regr, x, y, cv=4, scoring='neg_mean_squared_error')
                scores[(alpha, l1_ratio, normalize)] = np.mean(score)

    opt_params = max(scores, key=scores.get)
    alpha = opt_params[0]
    l1_ratio = opt_params[1]
    normalize = opt_params[2]

    return alpha, l1_ratio, normalize


def feature_selection(regr, train):
    x, y = train.drop(columns=['Id', 'SalePrice']), train['SalePrice']

    regr.fit(x, y)

    sfs = SFS(regr, k_features=x.shape[1] - 10, forward=False, verbose=2,
              scoring='neg_mean_squared_error', cv=4)
    sfs.fit(x, y)
    selected_features = (pd.DataFrame(sfs.get_metric_dict())
                         .T
                         .loc[:, ['feature_names', 'avg_score', 'std_dev', 'std_err']]
                         .sort_values(['avg_score', 'std_dev'], ascending=False)
                         .reset_index(drop=True))

    best_features = selected_features.at[0, 'feature_names']
    best_features = list(best_features)
    bad_features = [f for f in x if f not in best_features]

    return bad_features


def run():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    train, test = preprocessing(train, test)

    alpha, l1_ratio, normalize = tune_regression_params(train)

    regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, normalize=normalize)

    bad_features = feature_selection(regr, train)

    train = train.drop(columns=bad_features)
    test = test.drop(columns=bad_features)

    alpha, l1_ratio, normalize = tune_regression_params(train)

    regr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, normalize=normalize)

    x, y = train.drop(columns=['Id', 'SalePrice']), train['SalePrice']
    regr.fit(x, y)
    test['SalePrice'] = regr.predict(test.drop(columns=['Id']))

    result_funct(test, 'ElasticNet')

    joblib.dump(regr, 'ElasticNet.pkl')

