import base64
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector
import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route("/")
def hello_world():
    return jsonify({"hello": "world!"})

@app.route('/', methods=['POST'])
@cross_origin(supports_credentials=True)
def detect_hands():
    # get the image data from the request
    image_data = request.json['image']
    
    # convert the base64-encoded image data to a numpy array
    image_np = np.fromstring(base64.b64decode(image_data), dtype=np.uint8)
    
    # decode the numpy array as an OpenCV image
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    
    # initialize the detector
    detector = HandDetector(detectionCon=0.05, maxHands=2)
    faceDetector = FaceDetector()
    faceMeshDetector = FaceMeshDetector()
    
    # detect the hands in the image
    hands, img = detector.findHands(img)
    # print(str(base64.b64decode(img)))
    print("hands", len(hands))
    
    img, faceMesh = faceMeshDetector.findFaceMesh(img, draw=False) # 
    print("faceMesh", len(faceMesh))
    
    img, faces = faceDetector.findFaces(img) #
    print("faces", len(faces))
    
    # cv2.imwrite("temp_img.jpg", img)
    
    # return the results as a JSON object
    if hands:
        return jsonify({'success': False, 'status': 400, 'message': "Hands detect"}),400
    else:
        return jsonify({'success': True, 'status': 200, 'message': 'All good!'}),200

if __name__ == '__main__':
    app.run(debug=True, port=9000)