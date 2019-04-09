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





## Loops
result=np.zeros(test.shape[0])

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=25,random_state=10)
for counter,(train_index, valid_index) in enumerate(rskf.split(train, train.target),1):
    print (counter)
    
    #Train data
    t=train.iloc[train_index]
    t=augment(t)
    trn_data = lgb.Dataset(t.drop("target",axis=1), label=t.target)
    
    #Validation data
    v=train.iloc[valid_index]
    val_data = lgb.Dataset(v.drop("target",axis=1), label=v.target)
    
    #Training
    model = lgb.train(param, trn_data, 1000000, valid_sets = [val_data], verbose_eval=500, early_stopping_rounds = 3000)
    result += model.predict(test)
    
    
    
submission = pd.read_csv('data/santander-customer-transaction-prediction/sample_submission.csv')
submission['target'] = result/counter
submission.to_csv("LightGBM_1NoDataAugExtraFeatFEWERCV.csv", index=False)