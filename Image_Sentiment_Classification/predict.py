import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import sys
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
x_train_path = sys.argv[1]
y_train_path = sys.argv[2]
model_save_path = sys.argv[3]

model = load_model(model_save_path)
x_train = np.load(x_train_path) / 255.0
y_train = np.load(y_train_path).reshape(len(x_train)).astype('int')
y_to_cat = to_categorical(y_train)
x_train = x_train.reshape(len(x_train), 48, 48, 1)

y_pred= model.predict_classes(x_train)

model.evaluate(x_train,y_to_cat)


x=['1','2','3','4','5','6','7']
y=x
confmat = confusion_matrix(y_true=y_train, y_pred=y_pred )
fig, ax = plt.subplots()
ax.imshow(confmat, cmap=plt.cm.Blues)

ax.set_xticks(np.arange(7))
ax.set_yticks(np.arange(7))

ax.set_xticklabels(x)
ax.set_yticklabels(y)

for i in range(7):
    for j in range(7):
        ax.text(j, i, confmat[i][j],ha='center',va='center')

ax.set_title("confusion matrix")
fig.tight_layout()
plt.show()

