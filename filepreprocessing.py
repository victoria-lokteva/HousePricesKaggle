import numpy as np
import pandas as pd


def preprocessing(train, test):
    train['SalePrice'] = np.log(train['SalePrice'])
    data = train.append(test, sort=True)

    data['LotFrontage'] = data['LotFrontage'].groupby(by=data['Neighborhood']).transform(lambda x: x.fillna(x.median()))
    column_with_nans = data.columns[data.isna().any()].to_list()
    column_with_nans.remove('SalePrice')
    data[column_with_nans] = data[column_with_nans].fillna(0)
    data = ratings_to_numbers(data)

    cat_features = data.select_dtypes(include='object').columns.to_list()
    cat_features =  cat_features + ['MSSubClass']
    data = pd.get_dummies(data, columns= cat_features)

    bad_features = ['MoSold', '3SsnPorch', 'WoodDeckSF']
    data = data.drop(columns=bad_features)

    test = data.loc[data['SalePrice'].isna()].drop(columns=['SalePrice'])
    train = data.loc[data['SalePrice'].notna()]
    return train, test


def preprocessing_lgb(train, test):
    train['SalePrice'] = np.log(train['SalePrice'])
    data = train.append(test, sort=True)
    data['LotFrontage'] = data['LotFrontage'].groupby(by=data['Neighborhood']).transform(lambda x: x.fillna(x.median()))

    column_with_nans1 = ['MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                         'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea', 'GarageCars', 'Utilities']
    column_with_nans2 = ['Alley', 'PoolQC', 'MiscFeature', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType',
                         'MasVnrType', 'Fence',
                         'FireplaceQu', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual',
                         'Electrical']

    data[column_with_nans1] = data[column_with_nans1].fillna(0)
    data[column_with_nans2] = data[column_with_nans2].fillna('null')
    data = ratings_to_numbers(data)

    cat_features = function_cat(train)
    for feature in cat_features:
        data[feature] = data[feature].astype('category')

    test = data.loc[data['SalePrice'].isna()].drop(columns=['SalePrice'])
    train = data.loc[data['SalePrice'].notna()]
    return train, test


def function_cat(train):
    cat_features = train.select_dtypes(include='object').columns.to_list()
    cat_features = cat_features + ['MSSubClass']
    return cat_features


def result_funct(test, title):
    test['SalePrice'] = np.exp(test['SalePrice'])
    result = test.loc[:, ['Id', 'SalePrice']]
    title = str(title)
    result.to_csv('House_Price' + title + '.csv', index=False)


def ratings_to_numbers(data):
    d1 = {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1, 0: 0}
    data['Utilities'] = data['Utilities'].map(lambda x: d1[x])

    list1 = ['BsmtQual', 'BsmtCond', 'PoolQC', 'FireplaceQu', 'ExterQual', 'HeatingQC', 'KitchenQual',
             'GarageQual', 'GarageCond', 'ExterCond']
    d1 = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    for feature in list1:
        data[feature] = data[feature].map(lambda x: d1[x] if x in d1 else 0)

    d1 = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1}
    data['BsmtFinType1'] = data['BsmtFinType1'].map(lambda x: d1[x] if x in d1 else 0)
    data['BsmtFinType2'] = data['BsmtFinType2'].map(lambda x: d1[x] if x in d1 else 0)

    d1 = {'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0.5}
    data['Functional'] = data['Functional'].map(lambda x: d1[x] if x in d1 else 0)

    d1 = {'Fin': 4, 'RFn': 3, 'Unf': 2, 'No': 1}
    data['GarageFinish'] = data['GarageFinish'].map(lambda x: d1[x] if x in d1 else 0)

    d1 = {'Y': 3, 'P': 2, 'N': 1}
    data['PavedDrive'] = data['PavedDrive'].map(lambda x: d1[x] if x in d1 else 0)

    d1 = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1}
    data['BsmtExposure'] = data['BsmtExposure'].map(lambda x: d1[x] if x in d1 else 0)

    return data
