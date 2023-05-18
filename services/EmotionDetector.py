import base64
from pathlib import Path
import cv2
import os
import numpy as np
# from keras.models import load_model, Sequential
# from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
from os.path import join, dirname
import gdown
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )


class EmotionDetector:
    def __init__(self):
        print("EmotionDetector")
        self.__image = None
        # self.__detector = load_model(join(dirname(__file__), "emotionModel.hdf5"), compile=False)
        self.__labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        
    def detect(self, image):
        self.__image = image
        emotionTargetSize = self.__detector.input_shape[1:3]
        image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, emotionTargetSize)
        # img_gray = np.expand_dims(img_gray, axis=0)
        
        grayFace = img_gray.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)

        # emotion_predictions = self.__detector.predict(grayFace)
        # sum_of_predictions = emotion_predictions.sum()
        
        # obj = {}
        # obj["emotion"] = {}

        # for i, emotion_label in enumerate(self.__labels):
        #     emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
        #     obj["emotion"][emotion_label] = emotion_prediction

        # obj["dominant_emotion"] = self.__labels[np.argmax(emotion_predictions)]
        # print(obj)

    def loadModel(self, url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"):

        num_classes = 7

        model = Sequential()

        # 1st convolution layer
        model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # fully connected neural networks
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(num_classes, activation="softmax"))

        # ----------------------------

        home = str(Path.home())

        if os.path.isfile(home + "/.weights/facial_expression_model_weights.h5") != True:
            # print("facial_expression_model_weights.h5 will be downloaded...")

            output = home + "/.weights/facial_expression_model_weights.h5"
            gdown.download(url, output, quiet=True)

        model.load_weights(home + "/.weights/facial_expression_model_weights.h5")

        return model