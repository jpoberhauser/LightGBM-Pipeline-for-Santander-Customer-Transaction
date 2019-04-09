# santander_customer_transaction


https://www.kaggle.com/c/santander-customer-transaction-prediction



Code for entry in the Santander customer transaction prediction Kaggle competition




## Concepts tried:

* Data augmentation

concept: add small permutations to each row of data to give the model more data to train on
props to: https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment

* Automatic Feature Engineering

concept: some of the features (all are anonimized) _seem_ to be categorical. By adding extra features that are rounded versions of the continious vars, I get a better score on the public LB.

* Stratified Cross validation

* Target encoding with startified Kfold

* Model stacking and blending

* Xgboost


## best result

* LightGBM

```
params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 5,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.2,
    'subsample': 0.6,
    'eta': 0.01,
    'gamma': 0.65,
    'num_boost_round' : 700,
    'tree_method':'approx',
     "eval_metric": "auc"
    }
```



* 