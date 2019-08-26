import numpy as np
from tensorflow.keras.models import load_model
import sys
model_path = sys.argv[1]
predict_img_path = sys.argv[2]


model = load_model(model_path)

for i in range(7):
    img= np.load(f'{predict_img_path}/{i}.npy')
    print(model.predict(img))
