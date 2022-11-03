import tensorflow as tf
import numpy as np
import cv2

# Demonstration of using an already trained model for the purposes of predictions

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load in model
model = tf.keras.models.load_model('save/my_model')

probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

image_path = input('Enter image file path: ')

# Resize prediction image to proper dimensions and convert to greyscale
image = cv2.imread(image_path)
image = cv2.resize(image, (28, 28))
image = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)])


# print model's prediction
predictions = probability_model.predict(image)
print(class_names[np.argmax(predictions[0])])
