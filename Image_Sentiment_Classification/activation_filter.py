import sys
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
model_path = sys.argv[1]
tf.compat.v1.disable_eager_execution()
img_shape = (1,48,48,1)
lr =1
step = 100
model = load_model(model_path)


def normalization(x):
    return x/(k.sqrt(k.mean(k.square(x))) + 1e-7)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # print(x.shape)
    return x
def conv_layer_filter(model, img_shape, step, lr):
    layers_dict =dict([(layer.name, layer)  for layer in model.layers if 'con' in layer.name])
    for layer_name,layer in layers_dict.items():
        img_root_path = f'filter_img/{layer_name}'
        if not os.path.exists(img_root_path):
            os.mkdir(img_root_path)
        
        layer_output = layer.output
        input_img = model.input
        filter_num = layer.get_weights()[1].shape[0]
        for filter_index in range(filter_num):
            loss = k.mean(layer_output[:,:,:,filter_index])
            grads = k.gradients(loss, input_img)[0]
            grads = normalization(grads)
            iteration = k.function([input_img ], [loss, grads])
        
    
            input_img_data = np.random.random(img_shape)*20 + 128
            for i in range(step):
                loss_value, grads_value = iteration([input_img_data])
                input_img_data += grads_value * lr
            print('grad',grads_value.max())
            print(loss_value)
    
            img = input_img_data[0]
            img = deprocess_image(img).reshape(img_shape[1],img_shape[2])
    
            plt.imshow(img, cmap=plt.cm.gray)
            plt.savefig(f'{img_root_path}/filter-{filter_index}.png')
def output_layer_filter(model, img_shape, step, lr, output_num):
    for i in range(output_num):
        output_path = f'filter_img/output'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        input_img=model.input
        loss = k.mean(model.output[:,i])
        grads = k.gradients(loss,input_img)[0]
        grads = normalization(grads)
        iteration = k.function([input_img],[loss,grads])
    
        input_img_data = np.random.random(img_shape)
        for j in range(step):
            loss_value, grads_value = iteration([input_img_data])
            input_img_data += grads_value*lr 
        print(loss_value)

        np.save(f'{output_path}/{i}.npy',input_img_data)
        img = input_img_data[0]
        img = deprocess_image(img)
        img = img.reshape(img_shape[1],img_shape[2])
        plt.imshow(img, cmap=plt.cm.gray)

        plt.savefig(f'{output_path}/{i}.png')



if __name__ =='__main__':
    output_layer_filter(model,(1,48,48,1),100,0.02,7)
    # conv_layer_filter(model,(1,48,48,1),100,1)
