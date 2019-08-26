# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, LeakyReLU
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import sys

x_train_path = sys.argv[1]

y_train_path = sys.argv[2]


model_save_path = sys.argv[3]

x_train = np.load(x_train_path) / 255.0
y_train = np.load(y_train_path)


x_train = x_train.reshape(len(x_train), 48, 48, 1)
y_train = to_categorical(y_train)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, random_state=0)

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same',
                 input_shape=(48, 48, 1)))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(
    Conv2D(
        filters=32,
        kernel_size=(
            3,
            3),
        padding='same'
        ))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(
    Conv2D(
        filters=64,
        kernel_size=(
            3,
            3),
        padding='same'
        ))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

model.add(
    Conv2D(
        filters=128,
        kernel_size=(
            3,
            3),
        padding='same'
        ))
model.add(LeakyReLU(alpha=0.03))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.3))

# model.add(Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(units=256, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=512, activation='relu'))
# model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=7, activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


# featurewise_center = False, featurewise_std_normalization=False,

img = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False,
    dtype=np.float32)
img.fit(x_train)


# call back


callbacks = []
file_path = model_save_path
modelCheckpoint = ModelCheckpoint(
    file_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    save_weights_only=False)
callbacks.append(modelCheckpoint)


earlystop = EarlyStopping(monitor='val_loss', patience=6)

history = model.fit_generator(
    img.flow(
        x_train,
        y_train,
        batch_size=300),
    steps_per_epoch=600,
    epochs=100,
    validation_data=(
        x_val,
        y_val),
    callbacks=callbacks)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train_loss', 'vali_loss'])
plt.show()


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train_acc', 'val_acc'])
plt.show()
