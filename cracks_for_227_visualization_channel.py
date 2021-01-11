from keras.models import load_model
import numpy as np
from keras_preprocessing import image
import matplotlib.pyplot as plt
from keras import models

# load_model
model = load_model('/data-tmp/cracks/model/result_model/cracks_for_227.h5')

# check the model
model.summary()

# process the picture
img_path = '/data-tmp/cracks/datasets/row_data/Positive/05000.jpg'
img = image.load_img(img_path, target_size=(227, 227))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

layers_outputs = [layer.output for layer in model.layers[:7]]
activation_model = models.Model(inputs=model.input, outputs=layers_outputs)

activations = activation_model.predict(img_tensor)

# find the feature of activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()
plt.matshow(first_layer_activation[0, :, :, 6], cmap='viridis')
plt.show()

# show all
i = 1
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]

    size = layer_activation.shape[1]

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size,
                         row * size: (row + 1) * size] = channel_image

            scale = 1. / size
            plt.figure(figsize=(scale * display_grid.shape[1],
                                scale * display_grid.shape[0]))
            plt.grid(False)
            i += 1
    plt.title(layer_name)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig('/data-tmp/cracks/model/result_version/{}.pdf'.format(layer_name))

print(i)