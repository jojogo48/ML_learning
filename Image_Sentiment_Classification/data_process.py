import numpy as np
import pandas as pd
import sys



X_train_path = sys.argv[1]
x_train_save_path = './data/x_train'
y_train_save_path = './data/y_train'

X_train = pd.read_csv(X_train_path, delimiter=',')


X_train['feature'] = X_train['feature'].str.split(' ')

X_train = X_train.to_numpy()
x_train = np.zeros(shape=(X_train.shape[0], len(X_train[0][1])))
y_train = X_train[:,0].reshape(X_train.shape[0], 1)

for i in range(X_train.shape[0]):
    ls = X_train[i,1]                
    x_train[i] = np.array(ls, dtype='float32')


np.save(x_train_save_path, x_train)
np.save(y_train_save_path, y_train)

