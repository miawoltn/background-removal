import base64
import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
# from imageai.Detection import ObjectDetection
from deepface import DeepFace
from services.EmotionDetector import EmotionDetector

from services.GlassesDetector import GlassesDetector


import os
import gdown
import tensorflow as tf

# -------------------------------------------
# pylint: disable=line-too-long
# -------------------------------------------
# dependency configuration
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
elif tf_version == 2:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D,
        MaxPooling2D,
        AveragePooling2D,
        Flatten,
        Dense,
        Dropout,
    )

class DetectionService:
    """
    Detection service for detecting face, emotions and face wear such as headwear and glasses.
    The service also detects hands. It's likely this service will change in the future due to changes in client requirements.
    """
    def __init__(self):
        # initialize the detector
        
        # CVZone instantiation
        self.__faceDetector = FaceDetector()
        self.__faceMeshDetector = FaceMeshDetector()
        self.__handDetector = HandDetector(detectionCon=0.05, maxHands=2)
        
        # ImageAI instantiation
        # self.__objectDetecor = ObjectDetection()
        # self.__objectDetecor.setModelTypeAsRetinaNet()
        # self.__objectDetecor.setModelPath("/Users/aminuabdulmalik/Documents/Official/FACE_DETECTION_API/models/retinanet_resnet50_fpn_coco-eeacb38b.pth")
        # self.__objectDetecor.loadModel()
        
        # Glasses detector instantiation
        self.__glassesDetector = GlassesDetector()
        self.__emotionDetector = EmotionDetector()
        
        self.__image = None
        self.__features = {"all": self.__apply_detections, "face": self.__detect_face, "hand": self.__detect_hand, "glasses": self.__detect_glasses, "expression": self.__detect_emotions}
        
        self.detections = {}
        self.is_face = False
        self.is_multiple_faces = False
        self.has_hand = False
        self.has_headwear = False
        self.is_smiling = False
        self.is_sad = False
        self.is_angry = False
        self.is_surprised = False
        self.is_neutral = False
        self.has_glasses = False
        
    def detect(self, image, feature="all"):
        """
        Detect specified features in the image.
        :param image: Image to detect features from.
        :param feature: Feature to be detected in the image. Values can either be all, face, hand, glasses or expressions.
        """
        
        if not image: 
            raise Exception("Image is empty.")
        if feature not in self.__features:
            raise Exception("Feature does not exist.")
            
        self.__image = image
        detect_feature = self.__features[feature]
        detect_feature()
            
        
    def __apply_detections(self):
        """
        Applies multiple detections to the image
        """ 
        
        self.__detect_face()
        self.__detect_hand()
        self.__detect_glasses()
        self.__detect_emotions()
    
    def __detect_face(self):
        """
        Detects face in the image.
        """ 
        
        # convert the base64-encoded image data to a numpy array
        image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        img, faceMesh = self.__faceMeshDetector.findFaceMesh(img, draw=False) # 
        print("faceMesh", len(faceMesh))
        
        img, faces = self.__faceDetector.findFaces(img) #
        print("faces", len(faces))
        
        # cv2.imwrite("temp_img.jpg", img)
        
        self.is_face =  len(faces) == 1 
        self.is_multiple_faces = len(faces) > 1   
        
    def __detect_hand(self):
        """
        Detects hand in the image.
        """ 
        
        # convert the base64-encoded image data to a numpy array
        image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                
        # detect the hands in the image
        hands, img = self.__handDetector.findHands(img)
        
        print("hand: ", len(hands))
        
        # cv2.imwrite("temp_img.jpg", img)
        
        self.has_hand = len(hands) > 0 
        
    def __detect_glasses(self):
        """
        Detects glasses in the image.
        """ 
        self.has_glasses = self.__glassesDetector.hasGlasses(self.__image)
        
    def __detect_emotions(self):
        self.__emotionDetector.detect(self.__image)
        # convert the base64-encoded image data to a numpy array
        # image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        # img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        # result = DeepFace.analyze(img,actions=['emotion'], silent=True, detector_backend="mediapipe")
        # print(result)
        # self.is_neutral = result[0]['dominant_emotion'] == 'neutral'
        
        # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_gray = cv2.resize(img_gray, (48, 48))
        # img_gray = np.expand_dims(img_gray, axis=0)

        # emotion_predictions = self.loadModel().predict(img_gray, verbose="auto", use_multiprocessing=True)[0, :]

        # sum_of_predictions = emotion_predictions.sum()
        # obj = {}
        # obj["emotion"] = {}

        # labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        # for i, emotion_label in enumerate(labels):
        #     emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
        #     obj["emotion"][emotion_label] = emotion_prediction

        # obj["dominant_emotion"] = labels[np.argmax(emotion_predictions)]
        # print(obj)
        
    # def __detect_objects(self):
    #       # convert the base64-encoded image data to a numpy array
    #     image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
    #     # decode the numpy array as an OpenCV image
    #     img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    #     detections = self.__objectDetecor.detectObjectsFromImage(img, "img.jpg", minimum_percentage_probability=30)
    #     for eachObject in detections:
    #         print(eachObject["name"] , " : ", eachObject["percentage_probability"])
    #         print("--------------------------------")
       
    def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5",
    ):

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

        home = "/Users/aminuabdulmalik" #functions.get_deepface_home()

        if os.path.isfile(home + "/.deepface/weights/facial_expression_model_weights.h5") != True:
            print("facial_expression_model_weights.h5 will be downloaded...")

            output = home + "/.deepface/weights/facial_expression_model_weights.h5"
            gdown.download(url, output, quiet=False)

        model.load_weights(home + "/.deepface/weights/facial_expression_model_weights.h5")

        return model 
    