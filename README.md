# Project: Using Automatic Feature Engineering and XGBoost Ensembles to Predict Santander Customer Transactions

Full project with training and evaluation pipeline to predict which customers whil make which transactions in the future. 



https://www.kaggle.com/c/santander-customer-transaction-prediction



Code for entry in the Santander customer transaction prediction Kaggle competition.



## Concepts used:

* **Data augmentation**

Idea: Add small permutations to each row of data to give the model more data to train on. This is a variation of data augmentation techniques traditionally used in Computer Vision applied to tabular data. 

Code here: https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment

* **Automatic Feature Engineering**

Idea: The data given for this porkect is anonimized, and there is a possibility that some of these features are in fact categorical. We can add extra features which consist of the numerical features but rounded to different decimal points and see if I can get the true category. By adding extra features that are rounded versions of the continious vars, I get a better score on the public LB.

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
