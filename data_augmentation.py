# data augmentation
import pandas as pd
train = pd.read_csv('data/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('data/santander-customer-transaction-prediction/test.csv')



features = [c for c in train.columns if c not in ['ID_code', 'target']]
for feature in features:
    train['r2_'+feature] = np.round(train[feature], 2)
    test['r2_'+feature] = np.round(test[feature], 2)
    train['r1_'+feature] = np.round(train[feature], 1)
    test['r1_'+feature] = np.round(test[feature], 1)
    
    
    
train.to_csv("data/santander-customer-transaction-prediction/train_extra_features.csv")    
test.to_csv("data/santander-customer-transaction-prediction/test_extra_features.csv")    