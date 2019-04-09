## augment rows

## Inspiration from
#https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment


def augment(train,num_n=1,num_p=2):
    newtrain=[train]
    
    n=train[train.target==0]
    for i in range(num_n):
        newtrain.append(n.apply(lambda x:x.values.take(np.random.permutation(len(n)))))
    
    for i in range(num_p):
        p=train[train.target>0]
        newtrain.append(p.apply(lambda x:x.values.take(np.random.permutation(len(p)))))
    return pd.concat(newtrain)
#df=oversample(train,2,1)

