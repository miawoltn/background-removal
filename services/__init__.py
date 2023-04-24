import base64
import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector


class DetectionService:
    """
    Detection service for detecting face, emotions and face wear such as headwear and glasses.
    The service also detects hands. It's likely this service will change in the future due to changes in client requirements.
    """
    def __init__(self):
        # initialize the detector
        self.faceDetector = FaceDetector()
        self.faceMeshDetector = FaceMeshDetector()
        self.handDetector = HandDetector(detectionCon=0.05, maxHands=2)
        
        self.image = None
        self.detections = {}
        self.is_face = False
        self.is_multiple_faces = False
        self.has_hand = False
        self.has_headwear = False
        self.is_smiling = False
        self.is_sad = False
        self.is_angry = False
        self.is_surprised = False
        self.has_glasses = False
        
    def detect(self, image):
        if not image: 
            raise Exception("Image is null")
        self.image = image
        self.apply_detections()
        
    def apply_detections(self):
        self.__detect_face()
        self.__detect_hand()
    
    def __detect_face(self):
        # convert the base64-encoded image data to a numpy array
        image_np = np.fromstring(base64.b64decode(self.image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        
        img, faceMesh = self.faceMeshDetector.findFaceMesh(img, draw=False) # 
        print("faceMesh", len(faceMesh))
        
        img, faces = self.faceDetector.findFaces(img) #
        print("faces", len(faces))
        
        # cv2.imwrite("temp_img.jpg", img)
        
        self.is_face =  len(faces) == 1 
        self.is_multiple_faces = len(faces) > 1   
        
    def __detect_hand(self):
         # convert the base64-encoded image data to a numpy array
        image_np = np.fromstring(base64.b64decode(self.image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
                
        # detect the hands in the image
        hands, img = self.handDetector.findHands(img)
        
        print("hand: ", len(hands))
        
        # cv2.imwrite("temp_img.jpg", img)
        
        self.has_hand = len(hands) > 0 
        
    