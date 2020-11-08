import joblib
import pandas as pd
from mlxtend.regressor import StackingCVRegressor

from filepreprocessing import preprocessing, result_funct

ridge = joblib.load('Model_Reg_Ridge_opt.pkl')
ElasticNet = joblib.load('ElasticNet.pkl')
lightGBM1 = joblib.load('LGBM.pkl')

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
trainL, test = preprocessing(train, test)

blend = StackingCVRegressor(regressors=(ridge, lightGBM1, ElasticNet), meta_regressor=lightGBM1,
                            use_features_in_secondary=True, random_state=39)
blend_model = blend.fit(train.drop(columns=['Id', 'SalePrice']), train['SalePrice'])
test['SalePrice'] = blend_model.predict(test.drop(columns=['Id']))

test = result_funct(test, 'Blend')
