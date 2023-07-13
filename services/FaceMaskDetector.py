import base64
import bz2
import os
import cv2
import numpy as np
from imageai.Detection.Custom import CustomObjectDetection
from commons.functions import download_file, get_model_path

class FaceMaskDetector:
     def __init__(self):
        """
        This class detects face mask from a face
        """
        self.__model_path = f"{get_model_path()}/face_mask_model.pt"
        self.__config_path = f"{get_model_path()}/face_mask_config.json"
        print(self.__model_path)
        
        
     def __check_model(self):
        if os.path.isfile(self.__model_path) != True:
            model = "face_mask_model.pt"
            print(f"{model} is going to be downloaded")

            model_url = f"https://facemodules.s3.eu-central-1.amazonaws.com/models/{model}"

            download_file(model_url)
            print(f"{model} is downloaded")
        
        if os.path.isfile(self.__config_path) != True:
            config = "face_mask_config.json"
            print(f"{config} is going to be downloaded")
            
            config_url = f"https://facemodules.s3.eu-central-1.amazonaws.com/models/{config}"
            
            download_file(config_url)
            print(f"{config} is downloaded")
            
     def __init_detector(self):
         self.__check_model()
         self.__detector = CustomObjectDetection()
         self.__detector.setModelTypeAsYOLOv3()
         self.__detector.setModelPath(self.__model_path)
         self.__detector.setJsonPath(self.__config_path)
         self.__detector.loadModel()
        
     def detect(self, image):
        self.__image = image
        self.__init_detector()
        image_np = np.fromstring(base64.b64decode(self.__image), dtype=np.uint8)
        
        # decode the numpy array as an OpenCV image
        img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        detections = self.__detector.detectObjectsFromImage(input_image=img)
        print(detections)
        if len(detections) > 0:
            return detections[0]['percentage_probability'] > 50
        else: return False
        
        
        