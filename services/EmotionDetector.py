import base64
import cv2
import numpy as np
from deepface import DeepFace


class EmotionDetector:
    def __init__(self):
        print("EmotionDetector")
        self.__image = None

    def detect(self, image):
        self.__image = image
        image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        result = DeepFace.analyze(img,actions=['emotion'], silent=True)
        print(result[0])
        return result[0]['dominant_emotion']