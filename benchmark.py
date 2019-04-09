## benchmarks

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



## Perceptron
ppn = Perceptron(max_iter=5000, eta0=0.01, random_state=1)
ppn.fit(X_train_std, y_train)
y_pred = ppn.predict(X_valid_std)
roc_auc_score(y_valid, y_pred)



# Random Forest
forest = RandomForestClassifier(criterion='gini',
                                n_estimators=150, 
                                random_state=1,
                                n_jobs=-1)
forest.fit(X_train, y_train)
y_pred = forest.predict(X_valid)
roc_auc_score(y_valid, y_pred)

## simplest XGBOOST Model

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_valid, label=y_valid)
d_test = xgb.DMatrix(X_test)
watchlist = [(d_train,'train'), (d_valid,'valid')]
params = {
    'boosting':'dart',
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 3,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.5,
    'subsample': 0.8,
    'eta': 0.045,
    'gamma': 0.65,
    'num_boost_round' : 700,
    'tree_method':'approx',
     "eval_metric": "auc"
    }

mdl = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=100, maximize=True, verbose_eval=100)


pred = mdl.predict(d_test)
columnId = 'ID_code'
columnTarget = 'target'
sub = pd.DataFrame(test[columnId], columns=[columnId,columnTarget])
sub.head()
sub.to_csv('submitbenchmark_.csv'.format(mdl.__class__.__name__), index=False)