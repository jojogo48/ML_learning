import numpy as np
from LogisticRegression import LogisticRegression as LR


x_train = np.genfromtxt('./data/X_train', delimiter=',', skip_header=1)
y_train = np.genfromtxt('./data/Y_train', delimiter=',', skip_header=1)


model = LR()
model.load_data(x_train, y_train)
model.max_min_normal()
model.train(optimizer='Adagrad', lr=0.2, batch_size=len(x_train), epoch=1000)



