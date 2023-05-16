import base64
import cv2
import numpy as np
from keras.models import load_model
from os.path import join, dirname


class EmotionDetector:
    def __init__(self):
        print("EmotionDetector")
        self.__image = None
        self.__detector = load_model(join(dirname(__file__), "emotionModel.hdf5"), compile=False)
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