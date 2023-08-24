import base64
import cv2
import numpy as np
from PIL import Image
import io
from rembg import new_session, remove


class ImageBackground:
    def __init__(self):
        print("ImageBackground")
        self.__image = None

    def remove(self, image, color = [255, 255, 255, 255], output_format = "png"):
        self.__image = image
        img = Image.open(io.BytesIO(base64.b64decode(self.__image)))
        # print(img)
        
        result = remove(img, bgcolor = color)
        
        buffered = io.BytesIO()
        if output_format == 'png' or output_format is None:
            result.save(buffered, format="png")
        else:
            if result.mode in ("RGBA", "P"): result = result.convert("RGB")
            result.save(buffered, format="jpeg")
            
        string_result = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
        # print(string_result)
        return string_result