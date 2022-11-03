import tensorflow as tf
import numpy as np
import os

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# normalization step
train_images = train_images / 255.0
test_images = test_images / 255.0

SAVE_PATH = './save'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# Simple model with one hidden layer and one output layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Only 10 epochs so model trains quickly
model.fit(train_images, train_labels, epochs=10)

# Save the model so that it can be used in other programs for predictions
checkpoint = tf.train.latest_checkpoint(SAVE_PATH)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)


model.save('save/my_model')
