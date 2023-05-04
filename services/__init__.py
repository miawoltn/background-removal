import base64
import cv2
import numpy as np

from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
# from imageai.Detection import ObjectDetection

from services.GlassesDetector import GlassesDetector

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
        
        self.__image = None
        self.__features = {"all": self.__apply_detections, "hand": self.__detect_hand, "glasses": self.__detect_glasses, "expressions": None}
        
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
        
    def detect(self, image, feature="all"):
        """
        Detect specified features in the image.
        :param image: Image to detect features from.
        :param feature: Feature to be detected in the image.
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
        
    # def __detect_objects(self):
    #       # convert the base64-encoded image data to a numpy array
    #     image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
    #     # decode the numpy array as an OpenCV image
    #     img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    #     detections = self.__objectDetecor.detectObjectsFromImage(img, "img.jpg", minimum_percentage_probability=30)
    #     for eachObject in detections:
    #         print(eachObject["name"] , " : ", eachObject["percentage_probability"])
    #         print("--------------------------------")
        
    