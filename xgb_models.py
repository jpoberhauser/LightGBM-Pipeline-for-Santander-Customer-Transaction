## stratified Kfold Cross validation

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb

train = pd.read_csv('data/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('data/santander-customer-transaction-prediction/test.csv')
y = train.target
X = train.drop(['ID_code', 'target'], axis=1)
X_test = test.drop(['ID_code'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)





sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)


## Loops
for i, (train_index, test_index) in enumerate(skf.split(train, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    
    t = train.iloc[train_index]
    t = augment(t)   
    train_data = t.drop(['target','ID_code'], axis=1)
    y_train = t.target.values
    
    
    valid_data= train.iloc[test_index]
    y_valid = valid_data.target.values
    valid_data = valid_data.drop(['target','ID_code'], axis=1)
   
    # Convert our data into XGBoost format
    d_train = xgb.DMatrix(train_data, y_train)
    d_valid = xgb.DMatrix(valid_data, y_valid)
    d_test = xgb.DMatrix(X_test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    mdl = xgb.train(params, d_train, 15000, watchlist, early_stopping_rounds=300, maximize=True, verbose_eval=100)

    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    # Predict on our test data
    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
    sub['target'] += p_test/kfold
    
    
    
    
sub.columns = ["ID_code","target"]  
sub.to_csv("StratKFoldAugmentation.csv", index = None)