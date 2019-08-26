from tensorflow.keras.models import load_model
import numpy as np
import sys
import matplotlib.pyplot as plt

model_path = sys.argv[1]
x_train_path = sys.argv[2]
y_train_path = sys.argv[3]

model = load_model(model_path)

x_train = np.load(x_train_path)
y_train = np.load(y_train_path)


for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape, biases.shape)

fig,ax = plt.subplots(4,4)

filters, biases = model.layers[0].get_weights()
for i in range(4):
    for j in range(4):
        ai = ax[i,j]
        ai.set_xticks([0,1,2])
        ai.set_yticks([0,1,2])

        ai.set_xticklabels([0,1,2])
        ai.set_yticklabels([0,1,2])
        ai.imshow(filters[:,:,0,4*i+j], cmap=plt.cm.Blues) 
plt.show()
