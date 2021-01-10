from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

model = load_model('/data-tmp/cracks/model/result_model/cracks_for_227.h5')

layer_name = 'block3_conv1'
filter_index = 0

layout_output = model.get_layer(layer_name).output
loss = K.mean(layout_output[:, :, :, filter_index])

grads = K.gradients(loss, model.input)[0]
grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
iterate = K.function([model.input], [loss, grads])
loss_value, grads_value = iterate([np.zeros((1, 227, 227, 3))])
input_img_data = np.random.random((1, 227, 227, 3)) * 20 + 128.
step = 1

for i in range(40):
    loss_value, grads_value = iterate([input_img_data])

    input_img_data += grads_value * step


# Reprocess images
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# generate the pattern
def generate_pattern(layer_name, filter_index, size=227):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(loss, model.input)[0]

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.


    step =1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        img = input_img_data
        return deprocess_image(img)


plt.imshow(generate_pattern('block3_conv1', 0).reshape((227, 227, 3)))

block_num = [1, 2, 3, 4, 5]


def plot_filter(block_num):
    layer_name = 'block{}_conv1'.format(block_num)
    size = 64
    margin =5

    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):
        for j in range(8):
            plt.figure(figsize=(20, 20))
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start:horizontal_end,
                    vertical_start:vertical_end, :] = filter_img

    plt.figure(figszie=(20, 20))
    results /= 255
    plt.title('block{}_conv1'.format(block_num))
    plt.imshow(results)
    plt.savefig('/data-tmp/cracks/model/result_version/block{}_conv1.pdf'.format(block_num))

    return 0

for i in block_num:
    plot_filter(i)