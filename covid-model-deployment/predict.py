from io import BytesIO
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
IMG_SIZE = [227, 227]

class_names = ["COV", "NON-COV"]

model = None
probability_model = None


def load_model():
    model = tf.keras.models.load_model('model/my_model')
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    print("Model loaded")
    return model, probability_model


def predict(image):
    global model, probability_model
    if model is None:
        model, probability_model = load_model()

    nparr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (IMG_SIZE[0], IMG_SIZE[1]))
    image = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)])

    image = image/255.0
    #result = decode_predictions(model.predict(image), 2)[0]

    predictions = model.predict(image)
    print(predictions)
    predict_idx = (int)(predictions[0][0] > .5)
    print(predict)
    prob = 1-predictions[0][0]
    output = f'Prediction: {class_names[predict_idx]} Probability of covid: {prob:.2f}'

    return output


