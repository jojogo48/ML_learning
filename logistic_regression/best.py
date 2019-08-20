import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt


def normalization(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    return (x - x_mean)/x_std


X_train_path = sys.argv[1]
Y_train_path = sys.argv[2]


X_train = np.genfromtxt(X_train_path,delimiter=',',skip_header=1)
Y_train = np.genfromtxt(Y_train_path,delimiter=',',skip_header=1)

X_train = normalization(X_train)

x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, random_state=1)

model = Sequential()
model.add(Dense(input_dim=X_train.shape[1], units=4, activation='sigmoid'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=102, epochs=30, validation_data=(x_val, y_val))

train_loss = history.history['loss']
val_loss = history.history['val_loss']


train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['train','val'])
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.legend(['train','val'])
plt.show()





