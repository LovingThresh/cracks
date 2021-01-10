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
img_path = '/data-tmp/cracks/datasets/row_data_Positive/10000.jpg'

img = image.load_img(img_path, target_size=(227, 227))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

print(img_tensor.shape)

plt.imshow(img_tensor[0])
plt.show()

layers_outputs = [layer.output for layer in model.layers[:8]]
activation_model = model.Model(inputs=model.input, outputs=layers_outputs)

activations = activation_model.predict(img_tensor)

# find the feature of activation
first_layer_activation = activations[0]
print(first_layer_activation.shape)

plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')

# show all
